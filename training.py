import warnings
from torch import Tensor
from  typing import Tuple
from torch import nn,Union, Optional,Dict,Any
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from training_utils import AttributeDict,MetricsTracker,set_batch_count,display_and_save_batch,update_averaged_model,Eden,ScaledAdam,LRScheduler,Eve,get_scheduler,str2bool,load_checkpoint,fix_random_seed,save_checkpoint
from pathlib import Path
import logging
import random
import argparse
import copy
from shutil import copyfile
LRSchedulerType = torch.optim.lr_scheduler._LRScheduler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from models import add_model_arguments, get_model

def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    model_avg: nn.Module = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.

    If params.start_batch is positive, it will load the checkpoint from
    `params.exp_dir/checkpoint-{params.start_batch}.pt`. Otherwise, if
    params.start_epoch is larger than 1, it will load the checkpoint from
    `params.start_epoch - 1`.

    Apart from loading state dict for `model` and `optimizer` it also updates
    `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer that we are using.
      scheduler:
        The scheduler that we are using.
    Returns:
      Return a dict containing previously saved training info.
    """
    if params.start_batch > 0:
        filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
    elif params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    else:
        return None

    assert filename.is_file(), f"{filename} does not exist!"

    if isinstance(model, DDP):
        raise ValueError("load_checkpoint before DDP")

    saved_params = load_checkpoint(
        filename,
        model=model,
        model_avg=model_avg,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    saved_stage = saved_params.get("train_stage", 0)
    if params.train_stage != saved_stage:
        # switch training stage
        if params.train_stage and saved_stage:  # switch between 1 and 2
            params.start_epoch = 1
            params.start_batch = 0
        else:
            # switch between 0 and 1/2
            assert params.num_epochs >= params.start_epoch
            params.batch_idx_train = saved_params["batch_idx_train"]

        for key in ["optimizer", "grad_scaler", "sampler"]:
            if key in saved_params:
                saved_params.pop(key)

        # when base on stage 0, we keep scheduler
        if saved_stage != 0:
            for key in ["scheduler"]:
                if key in saved_params:
                    saved_params.pop(key)

        best_train_filename = params.exp_dir / "best-train-loss.pt"
        if best_train_filename.is_file():
            copyfile(
                src=best_train_filename,
                dst=params.exp_dir / f"best-train-loss-stage{saved_stage}.pt",
            )

        best_valid_filename = params.exp_dir / "best-valid-loss.pt"
        if best_valid_filename.is_file():
            copyfile(
                src=best_valid_filename,
                dst=params.exp_dir / f"best-valid-loss-stage{saved_stage}.pt",
            )
    else:

        keys = [
            "best_train_epoch",
            "best_valid_epoch",
            "batch_idx_train",
            "best_train_loss",
            "best_valid_loss",
        ]
        for k in keys:
            params[k] = saved_params[k]

        if params.start_batch > 0:
            if "cur_epoch" in saved_params:
                params["start_epoch"] = saved_params["cur_epoch"]

    return saved_params

def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    batch: dict,
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute transducer loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training. It is an instance of Zipformer in our case.
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
     warmup: a floating point value which increases throughout training;
        values >= 1.0 are fully warmed up and have all modules present.
    """
    device = (
        model.device
        if isinstance(model, DDP)
        else next(model.parameters()).device
    )
    # at entry, TextTokens is (N, P)
    text_tokens = batch["text_tokens"].to(device)
    text_tokens_lens = batch["text_tokens_lens"].to(device)
    assert text_tokens.ndim == 2

    audio_features = batch["audio_features"].to(device)
    audio_features_lens = batch["audio_features_lens"].to(device)
    assert audio_features.ndim == 3

    with torch.set_grad_enabled(is_training):
        predicts, loss, metrics = model(
            x=text_tokens,
            x_lens=text_tokens_lens,
            y=audio_features,
            y_lens=audio_features_lens,
            train_stage=params.train_stage,
        )

    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info["frames"] = (audio_features_lens).sum().item()
        info["utterances"] = text_tokens.size(0)

    # Note: We use reduction=sum while computing the loss.
    info["loss"] = loss.detach().cpu().item()
    for metric in metrics:
        info[metric] = metrics[metric].detach().cpu().item()
    del metrics

    return predicts, loss, info

def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        predicts, loss, loss_info = compute_loss(
            params=params,
            model=model,
            batch=batch,
            is_training=False,
        )
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info
    if world_size > 1:
        tot_loss.reduce(loss.device)
    loss_value = tot_loss["loss"] / tot_loss["frames"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    if params.visualize:
        output_dir = Path(
            f"{params.exp_dir}/eval/step-{params.batch_idx_train:06d}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(model, DDP):
            model.module.visualize(predicts, batch, output_dir=output_dir)
        else:
            model.visualize(predicts, batch, output_dir=output_dir)

    return tot_loss

def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer: torch.optim.Optimizer,
    scheduler: LRSchedulerType,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer we are using.
      scheduler:
        The learning rate scheduler, we call step() every step.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      rng:
        Random for selecting.
      scaler:
        The scaler used for mix precision training.
      model_avg:
        The stored model averaged from the start of training.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
      rank:
        The rank of the node in DDP training. If no DDP is used, it should
        be set to 0.
    """
    model.train()
    tot_loss = MetricsTracker()
    iter_dl = iter(train_dl)
    print(iter_dl)
    dtype, enabled = torch.float32, False
    if params.dtype in ["bfloat16", "bf16"]:
        dtype, enabled = torch.bfloat16, True
    elif params.dtype in ["float16", "fp16"]:
        dtype, enabled = torch.float16, True

    batch_idx = 0
    while True:
        try:
            batch = next(iter_dl)
            print(batch)
        except StopIteration:
            logging.info("Reaches end of dataloader.")
            break

        batch_idx += 1

        params.batch_idx_train += 1
        batch_size = len(batch["text"])

        try:
            with torch.cuda.amp.autocast(dtype=dtype, enabled=enabled):
                _, loss, loss_info = compute_loss(
                    params=params,
                    model=model,
                    batch=batch,
                    is_training=True,
                )
            # summary stats
            tot_loss = (
                tot_loss * (1 - 1 / params.reset_interval)
            ) + loss_info * (1 / params.reset_interval)

            # NOTE: We use reduction==sum and loss is computed over utterances
            # in the batch and there is no normalization to it so far.

            scaler.scale(loss).backward()
            if params.batch_idx_train >= params.accumulate_grad_steps:
                if (
                    params.batch_idx_train % params.accumulate_grad_steps
                    == 0
                ):
                    if params.optimizer_name not in ["ScaledAdam", "Eve"]:
                        # Unscales the gradients of optimizer's assigned params in-place
                        scaler.unscale_(optimizer)
                        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 1.0
                        )

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    for k in range(params.accumulate_grad_steps):
                        if isinstance(scheduler, Eden):
                            scheduler.step_batch(params.batch_idx_train)
                        else:
                            scheduler.step()

            set_batch_count(model, params.batch_idx_train)
        except:  # noqa
            display_and_save_batch(batch, params=params)
            raise

        if params.average_period > 0:
            if (
                params.batch_idx_train > 0
                and params.batch_idx_train % params.average_period == 0
            ):
                # Perform Operation in rank 0
                if rank == 0:
                    update_averaged_model(
                        params=params,
                        model_cur=model,
                        model_avg=model_avg,
                    )
             
      
         
        if batch_idx % 100 == 0 and params.dtype in ["float16", "fp16"]:
            # If the grad scale was less than 1, try increasing it.    The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have different
            # behavior depending on the current grad scale.
            cur_grad_scale = scaler._scale.item()
            if cur_grad_scale < 1.0 or (
                cur_grad_scale < 8.0 and batch_idx % 400 == 0
            ):
                scaler.update(cur_grad_scale * 2.0)

            if cur_grad_scale < 0.01:
                logging.warning(f"Grad scale is small: {cur_grad_scale}")
            if cur_grad_scale < 1.0e-05:
                raise RuntimeError(
                    f"grad_scale is too small, exiting: {cur_grad_scale}"
                )

        if batch_idx % params.log_interval == 0:
            cur_lr = scheduler.get_last_lr()[0]
            cur_grad_scale = (
                scaler._scale.item()
                if params.dtype in ["float16", "fp16"]
                else 1.0
            )

            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, train_loss[{loss_info}], "
                f"tot_loss[{tot_loss}], "
                f"batch size: {batch_size}, "
                f"lr: {cur_lr:.2e}"
                + (
                    f", grad_scale: {cur_grad_scale}"
                    if params.dtype in ["float16", "fp16"]
                    else ""
                )
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )
                loss_info.write_summary(
                    tb_writer,
                    "train/current_",
                    params.batch_idx_train,
                )
                tot_loss.write_summary(
                    tb_writer, "train/tot_", params.batch_idx_train
                )
                tot_loss.write_summary(
                    tb_writer, "train/tot_", params.batch_idx_train
                )
                if params.dtype in ["float16", "fp16"]:
                    tb_writer.add_scalar(
                        "train/grad_scale",
                        cur_grad_scale,
                        params.batch_idx_train,
                    )

        if params.batch_idx_train % params.valid_interval == 0:
            # Calculate validation loss in Rank 0
            model.eval()
            logging.info("Computing validation loss")
            with torch.cuda.amp.autocast(dtype=dtype):
                valid_info = compute_validation_loss(
                    params=params,
                    model=model,
                    valid_dl=valid_dl,
                    world_size=world_size,
                )
            logging.info(
                f"Epoch {params.cur_epoch}, validation: {valid_info}"
            )
            logging.info(
                f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
            )

            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )

            model.train()

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss

def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - best_train_loss: Best training loss so far. It is used to select
                           the model that has the lowest training loss. It is
                           updated during the training.

        - best_valid_loss: Best validation loss so far. It is used to select
                           the model that has the lowest validation loss. It is
                           updated during the training.

        - best_train_epoch: It is the epoch that has the best training loss.

        - best_valid_epoch: It is the epoch that has the best validation loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - valid_interval:  Run validation if batch_idx % valid_interval is 0
    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 100,  # 10: debug 100: train
            "reset_interval": 200,
            "valid_interval": 10000,
        }
    )

    return params

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="""If positive, --start-epoch is ignored and
        it loads the checkpoint from exp-dir/checkpoint-{start_batch}.pt
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="exp/vallex_dev",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--optimizer-name",
        type=str,
        default="ScaledAdam",
        help="The optimizer.",
    )
    parser.add_argument(
        "--scheduler-name",
        type=str,
        default="Eden",
        help="The scheduler.",
    )
    parser.add_argument(
        "--base-lr", type=float, default=0.05, help="The base learning rate."
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=200,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--inf-check",
        type=str2bool,
        default=False,
        help="Add hooks to check for infinite module outputs and gradients.",
    )

    parser.add_argument(
        "--save-every-n",
        type=int,
        default=10000,
        help="""Save checkpoint after processing this number of batches"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train %% save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/checkpoint-{params.batch_idx_train}.pt'
        Note: It also saves checkpoint to `exp-dir/epoch-xxx.pt` at the
        end of each epoch where `xxx` is the epoch number counting from 0.
        """,
    )
    parser.add_argument(
        "--valid-interval",
        type=int,
        default=10000,
        help="""Run validation if batch_idx %% valid_interval is 0.""",
    )

    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=20,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=0,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )

    parser.add_argument(
        "--accumulate-grad-steps",
        type=int,
        default=1,
        help="""update gradient when batch_idx_train %% accumulate_grad_steps == 0.
        """,
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Training dtype: float32 bfloat16 float16.",
    )

    parser.add_argument(
        "--filter-min-duration",
        type=float,
        default=0.0,
        help="Keep only utterances with duration > this.",
    )
    parser.add_argument(
        "--filter-max-duration",
        type=float,
        default=20.0,
        help="Keep only utterances with duration < this.",
    )

    parser.add_argument(
        "--train-stage",
        type=int,
        default=0,
        help="""0: train all modules, For VALL-E, support 1: AR Decoder 2: NAR Decoder(s)
        """,
    )

    parser.add_argument(
        "--visualize",
        type=str2bool,
        default=False,
        help="visualize model results in eval step.",
    )

    parser.add_argument(
        "--oom-check",
        type=str2bool,
        default=True,
        help="perform OOM check on dataloader batches before starting training.",
    )

    add_model_arguments(parser)

    return parser

def run(model,train_loader,valid_loader,checkpoints):
    parser = get_parser()
    args = parser.parse_args()
    params = get_params()
    params.update(vars(args))

    fix_random_seed(params.seed)
    rng = random.Random(params.seed)
    logging.info("Training started")
    rank=0
    if args.tensorboard and rank == 0:
        if params.train_stage:
            tb_writer = SummaryWriter(
                log_dir=f"{params.exp_dir}/tensorboard_stage{params.train_stage}"
            )
        else:
            tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    logging.info(f"Device: {device}")
    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)
    with open(f"{params.exp_dir}/model.txt", "w") as f:

        print(model, file=f)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0 and params.average_period > 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)

    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    model.to(device)
    if params.train_stage:
        _model = model.module if isinstance(model, DDP) else model
        model_parameters = _model.stage_parameters(params.train_stage)
    else:
        model_parameters = model.parameters()

    if params.optimizer_name == "ScaledAdam":
        parameters_names = []
        if params.train_stage:  # != 0
            _model = model.module if isinstance(model, DDP) else model
            parameters_names.append(
                [
                    name_param_pair[0]
                    for name_param_pair in _model.stage_named_parameters(
                        params.train_stage
                    )
                ]
            )
        else:
            parameters_names.append(
                [
                    name_param_pair[0]
                    for name_param_pair in model.named_parameters()
                ]
            )

        optimizer = ScaledAdam(
            model_parameters,
            lr=params.base_lr,
            betas=(0.9, 0.95),
            clipping_scale=2.0,
            parameters_names=parameters_names,
            show_dominant_parameters=False,
            clipping_update_period=1000,
        )
    elif params.optimizer_name == "Eve":
        optimizer = Eve(
            model_parameters,
            lr=params.base_lr,
            betas=(0.9, 0.98),
            target_rms=0.1,
        )
    elif params.optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model_parameters,
            lr=params.base_lr,
            betas=(0.9, 0.95),
            weight_decay=1e-2,
            eps=1e-8,
        )
    elif params.optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model_parameters,
            lr=params.base_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
    else:
        raise NotImplementedError()

    scheduler = get_scheduler(params, optimizer)
    optimizer.zero_grad()

    if checkpoints and "optimizer" in checkpoints:
        logging.info("Loading optimizer state dict")
        optimizer.load_state_dict(checkpoints["optimizer"])

    if (
        checkpoints
        and "scheduler" in checkpoints
        and checkpoints["scheduler"] is not None
    ):
        logging.info("Loading scheduler state dict")
        scheduler.load_state_dict(checkpoints["scheduler"])

    
    """ if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
        sampler_state_dict = checkpoints["sampler"]
    else:
        sampler_state_dict = None
 """
    scaler = GradScaler(
        enabled=(params.dtype in ["fp16", "float16"]), init_scale=1.0
    )
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        if isinstance(scheduler, Eden):
            scheduler.step_epoch(epoch - 1)

        
        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_loader,
            valid_dl=valid_loader,
            scaler=scaler,
            tb_writer=tb_writer,
            rank=rank,
        )
        
        save_checkpoint(
            filename="vallex_ar_checkpoint.pt",
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            rank=rank,
        )
       

    logging.info("Done!")





