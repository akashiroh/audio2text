from torch.utils.data import Dataset
from torch import tensor
import pandas as pd

from tokenizers import Tokenizer, normalizers, pre_tokenizers
import torchaudio

from config import config
from pathlib import Path


class AudioDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        tokenizer,
        sample_rate: int=16000,
        transform=None,
    ):
        df = pd.read_parquet(data_path / "transcriptions.parquet")
        
        self.tokenized = []
        for transcription in df.transcription:
            self.tokenized.append(
                tokenizer.encode(transcription.upper())
            )

        self.waveforms = []
        for i, file in enumerate(df["id"]):
            file_name = Path(data_path) / "all_audio_files" / f"{file}.wav"
            waveform, sr = torchaudio.load(file_name, format="wav")
            if sr != sample_rate:
                waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)

            if transform:
                waveform = transform(waveform)

            self.waveforms.append(waveform)

        print(f"Audio Dataset | returned {len(df)} datapoints")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        _input = torch.tensor(self.tokenized[idx])
        _output = self.waveforms[idx]

        return _input, _output
