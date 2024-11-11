import torch
from torch.utils.data import DataLoader
import speechbrain as sb
from speechbrain.pretrained import EncoderDecoderASR

from dataset import AudioDataset
from utils import batch_collate_fn

from jiwer import wer
from pathlib import Path

from config import config

def pipeline():
    dataset = AudioDataset(Path(config.paths.data))
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=batch_collate_fn,
    )

    model = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-transformer-transformerlm-librispeech",
        savedir="pretrained_models/asr-transformer-transformerlm-librispeech"
    )

    breakpoint()




if __name__ == "__main__":
    pipeline()
