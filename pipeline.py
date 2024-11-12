import torch
from torch.utils.data import DataLoader
import speechbrain as sb
from speechbrain.inference.ASR import EncoderDecoderASR, WhisperASR, StreamingASR
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig

from dataset import AudioDataset
from utils import batch_collate_fn

import pandas as pd
from jiwer import wer
from pathlib import Path
from tqdm import tqdm as tqdm

from config import config

def pipeline():

    df = pd.read_parquet(Path(config.paths.data) / "transcriptions.parquet")

    # model = EncoderDecoderASR.from_hparams(
    #     source="speechbrain/asr-transformer-transformerlm-librispeech",
    #     savedir="pretrained_models/asr-transformer-transformerlm-librispeech"
    # )
    # model = StreamingASR.from_hparams(
    #     source="speechbrain/asr-streaming-conformer-librispeech",
    #     savedir="speechbrain/model",
    # )
    model = StreamingASR.from_hparams(
        source="speechbrain/asr-streaming-conformer-gigaspeech",
        savedir="speechbrain/model",
    )
    
    preds = []
    for audio_file in tqdm(df["id"]):
        pred = model.transcribe_file(
            f"{config.paths.data}/all_audio_files/{audio_file}.wav",
            DynChunkTrainConfig(24, 4),
            use_torchaudio_streaming=False,
        )
        preds.append(pred)
    
    error_rates = []
    running_error_rate = 0

    for (pred, transcription) in zip(df.transcription, preds):
        error_rate = wer(transcription.lower(), pred.lower())
        error_rates.append(error_rate)
        running_error_rate += error_rate
    print(f"Average Error Rate: {running_error_rate / len(preds)}")

    breakpoint()


if __name__ == "__main__":
    pipeline()
