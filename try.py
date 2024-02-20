""" #=========================== compute loss =====================#

def compute_loss(model, batch):
    device = next(model.parameters()).device
    text_tokens = batch["input_ids"].to(device)
    audio_features = batch["audio_features"].to(device)
    inputs = {"input_ids": text_tokens, "audio_features": audio_features}
    outputs = model(**inputs)
    loss = outputs.loss
    return loss

#============================= seq2seqData_collator ================#
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import numpy as np
#@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

         utt_id_s = [b.get('utt_id', None) for b in features]
         print("=== utt_id_s ===",utt_id_s)
         text_s = [b.get('text', None) for b in features]
         print("=== text_s ===",text_s)
         audio_s = [b.get('audio', None) for b in features]
         print("=== audio_s ===",audio_s)
         audio_lens_s = [b.get('audio_lens', None) for b in features]
    #print("=== utt_id_s ===",utt_id_s)
    
    # Filter out None values from audio_features_lens_s before computing maximum
         audio_features_lens_s = [b['audio_features_lens'] for b in features if b.get('audio_features_lens') is not None]
    
    # Compute maximum audio feature length if audio_features_lens_s is not empty
         max_audio_features_len = max(audio_features_lens_s) if audio_features_lens_s else 0
    
    # Filter out None values from text_tokens_lens_s before computing maximum
         text_tokens_lens_s = [b.get('text_tokens_lens', None) for b in features if b.get('text_tokens_lens') is not None]
    
    # Compute maximum text tokens length if text_tokens_lens_s is not empty
         max_text_tokens_len = max(text_tokens_lens_s) if text_tokens_lens_s else 0
    
    # Create an empty tensor with maximum audio feature length
         audio_features_s = torch.zeros([len(features), max_audio_features_len, 8], dtype=torch.int64) - 1  # audio pad with -1

    # Create an empty tensor with maximum text tokens length
         text_tokens_s = torch.zeros([len(features), max_text_tokens_len], dtype=torch.int64) + 3  # [PAD] token id 3

    # Extract language information from each sample in the batch
         language_s = [b.get('language', None) for b in features]
    
    # Filter out None values and convert language information to a list of integers
         language_s = [lang for lang in language_s if lang is not None]
    
    # Convert language_s to a tensor only if it's not empty
         if language_s:
             language_tensor = torch.LongTensor(language_s)
         else:
             language_tensor = torch.tensor([])
         return {
        'utt_id': utt_id_s,
        'text': text_s,
        'audio': audio_s,
        'audio_lens': audio_lens_s,
        'audio_features': audio_features_s,
        'audio_features_lens': torch.LongTensor(np.array(audio_features_lens_s)),
        'text_tokens': text_tokens_s,
        'text_tokens_lens': torch.LongTensor(np.array(text_tokens_lens_s)),
        'languages': language_tensor,
    } 
#data_collator=DataCollatorSpeechSeq2Seq(train_dataset)
#======================== finetuning model =================#

#data_collator = collate(train_dataset)
#============= print dataset ==================#for sample in train_dataset:
    print(sample) 



import glob
import os
data_dir="D:\\MachineCourse\\Graduation_Project\\VALL-E-X\\mgb2_dataset\\train"
audio_paths = glob.glob(os.path.join(f"{data_dir}\\wav", '*.wav'))
text_paths = glob.glob(os.path.join(f"{data_dir}\\txt", '*.txt')) """





#========================= trainer =======================#
import random
from typing import Optional
from xml.dom.minidom import AttributeList
from altair import Union
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import logging

from torchmetrics import MetricTracker
from bin.trainer import LRSchedulerType, compute_loss, compute_validation_loss, display_and_save_batch, set_batch_count

from modules.optim import Eden
from torch import nn


def train_one_epoch(
    params: AttributeList,
    model: Union[nn.Module, DDP],
    optimizer: torch.optim.Optimizer,
    scheduler: LRSchedulerType,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    rng: random.Random,
    scaler: GradScaler,
    model_avg: Optional[nn.Module] = None,
    #tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:

    model.train()
    tot_loss = MetricTracker()
    iter_dl = iter(train_dl)

    dtype, enabled = torch.float32, False
    if params.dtype in ["bfloat16", "bf16"]:
        dtype, enabled = torch.bfloat16, True
    elif params.dtype in ["float16", "fp16"]:
        dtype, enabled = torch.float16, True

    batch_idx = 0
    while True:
        try:
            batch = next(iter_dl)
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

      
         
            # Perform Operation in rank 0
          
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

            model.train()

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss

