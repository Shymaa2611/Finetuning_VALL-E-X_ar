import logging
import h5py
import glob
import torch
import numpy as np
import os
from torch import nn
from torch.optim import Adam
import torchaudio
import soundfile as sf
from macros import N_DIM, NUM_HEAD, NUM_LAYERS, NUM_QUANTIZERS, PREFIX_MODE
from utils.g2p.symbols import symbols
from utils.g2p import PhonemeBpeTokenizer
from data.collation import get_text_token_collater
from data.dataset import create_dataloader, AudioDataset,collate
from data.tokenizer import AudioTokenizer, tokenize_audio
from typing import Any, Dict, Optional, Tuple, Union,Mapping
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from models.vallex import VALLE
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import Trainer
from tokenizers import Tokenizer
from mgb2 import create_dataset_parts
from training import run
checkpoints_dir = "./checkpoints/"
model_checkpoint_name = "vallex-checkpoint.pt"


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def prepare_dataset(name, audio_prompt_path, transcript):
    text_tokenizer = PhonemeBpeTokenizer(tokenizer_path="./utils/g2p/bpe_69.json")
    text_collater = get_text_token_collater()
    codec = AudioTokenizer(device)
    wav_pr, sr = torchaudio.load(audio_prompt_path)
    if wav_pr.size(0) == 2:
        wav_pr = wav_pr.mean(0, keepdim=True)
    # add language 
    text_pr = "[AR]" + transcript + "[AR]"
     # audio tokenizer
    encoded_frames = tokenize_audio(codec, (wav_pr, sr))
    audio_tokens = encoded_frames[0][0].transpose(2, 1).cpu().numpy()
    # text tokenizer
    phonemes, langs = text_tokenizer.tokenize(text=f"{text_pr}".strip())
    text_tokens, enroll_x_lens = text_collater(
        [
            phonemes
        ]
    )

    return audio_tokens, text_tokens, langs, text_pr 
   
def create_dataset(data_dir, dataloader_process_only):
    if dataloader_process_only:
        h5_output_path = os.path.join(data_dir, 'audio_sum.hdf5')
        ann_output_path = os.path.join(data_dir, 'audio_ann_sum.txt')
        audio_paths = glob.glob(os.path.join(f"{data_dir}\\wav", '*.wav'))
        text_paths = glob.glob(os.path.join(f"{data_dir}\\txt", '*.txt'))
        count = 0
        transcript = None

        with h5py.File(h5_output_path, 'w') as h5_file:
            for audio_path in audio_paths:
                stem = os.path.splitext(os.path.basename(audio_path))[0]
                if count < len(text_paths):
                    with open(text_paths[count], 'r', encoding='utf-8') as f:
                        line = f.readline()
                        transcript = line
                audio_tokens, text_tokens, langs, text = prepare_dataset(name=stem, audio_prompt_path=audio_path, transcript=transcript)
                count += 1
                print("text tokens ----> ",text_tokens)
                text_tokens = text_tokens.squeeze(0)
                grp = h5_file.create_group(stem)
                grp.create_dataset('audio', data=audio_tokens)
                grp.create_dataset('text', data=text_tokens)
                
                with open(ann_output_path, 'a', encoding='utf-8') as ann_file:
                    try:
                        audio, sample_rate = sf.read(audio_path)
                        duration = len(audio) / sample_rate
                        ann_file.write(f'{stem}|{duration}|{langs[0]}|{text}\n') 
                        print(f"Successfully wrote to {ann_output_path}")
                    except Exception as e:
                        print(f"An error occurred: {e}")
    else:
        dataloader = create_dataloader(data_dir=data_dir)
        return dataloader 

def create_dataset_format():
    #train
    data_dir = "mgb2_dataset\\train"
    create_dataset(data_dir=data_dir, dataloader_process_only=True) 
    train_dataset = AudioDataset(h5_path=os.path.join(data_dir, 'audio_sum.hdf5'),
                             ann_path=os.path.join(data_dir, 'audio_ann_sum.txt'),
                             tokenizer_path="./utils/g2p/bpe_69.json")
    print("finish train  dataset !")
    #test
    data_dir = "mgb2_dataset\\test"
    create_dataset(data_dir=data_dir, dataloader_process_only=True) 
    test_dataset = AudioDataset(h5_path=os.path.join(data_dir, 'audio_sum.hdf5'),
                             ann_path=os.path.join(data_dir, 'audio_ann_sum.txt'),
                             tokenizer_path="./utils/g2p/bpe_69.json")
    print("finish test dataset !")
    #dev
    data_dir = "mgb2_dataset\\dev"
    create_dataset(data_dir=data_dir, dataloader_process_only=True) 
    dev_dataset = AudioDataset(h5_path=os.path.join(data_dir, 'audio_sum.hdf5'),
                             ann_path=os.path.join(data_dir, 'audio_ann_sum.txt'),
                             tokenizer_path="./utils/g2p/bpe_69.json")
    print("finish dev dataset !")
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=16,collate_fn=collate)
    valid_loader = torch.utils.data.DataLoader(test_dataset,batch_size=8,collate_fn=collate)
    return train_loader,valid_loader

#load pretrained model 
def load_model():
   if not os.path.exists(checkpoints_dir): os.mkdir(checkpoints_dir)
   if not os.path.exists(os.path.join(checkpoints_dir, model_checkpoint_name)):
        import wget
        try:
            logging.info(
                "Downloading model from https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt ...")
            wget.download("https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt",
                          out="./checkpoints/vallex-checkpoint.pt", bar=wget.bar_adaptive)
        except Exception as e:
            logging.info(e)
            raise Exception(
                "\n Model weights download failed, please go to 'https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt'"
                "\n manually download model weights and put it to {} .".format(os.getcwd() + "\checkpoints"))
    # VALL-E
   model = VALLE(
        N_DIM,
        NUM_HEAD,
        NUM_LAYERS,
        norm_first=True,
        add_prenet=False,
        prefix_mode=PREFIX_MODE,
        share_embedding=True,
        nar_scale_factor=1.0,
        prepend_bos=True,
        num_quantizers=NUM_QUANTIZERS,
    ).to(device)
   checkpoint = torch.load(os.path.join(checkpoints_dir, model_checkpoint_name), map_location='cpu')
   model.load_state_dict(checkpoint["model"])
   print("finish load pretrained model")
   return model,checkpoint


if __name__ == "__main__":
    dataset_dir=""
    create_dataset_parts(dataset_dir)
    train_loader,valid_loader=create_dataset_format()
    model,checkpoint=load_model()
    run(model,train_loader,valid_loader,checkpoint)
   
   











