from torch.utils.data import Dataset
import pandas as pd

from tokenizers import Tokenizer, normalizers, pre_tokenizers
import torchaudio

from pathlib import Path

class AudioDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        sample_rate: int=16000,
        transform=None,
    ):

        tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        df = pd.read_parquet(data_path / "transcriptions.parquet")
        
        self.tokenized = []
        for transcription in df.transcription:
            self.tokenized.append(
                tokenizer.encode(transcription)
            )

        self.waveforms = []
        for i, file in enumerate(df["id"]):
            file_name = f"../data/audio2text/all_audio_files/{file}.wav"
            waveform, sr = torchaudio.load(file_name, format="wav")
            if sr != sample_rate:
                waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)

            if transform:
                waveform = transform(waveform)

            self.waveforms.append(waveform)

        print(f"Audio Dataset | returned {len(df)} datapoints")
        breakpoint()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        transcription = self.tokenized[idx]
        waveform = self.waveforms[idx]

        _input = torch.tensor(transcription)
        _output = waveform1

        return _input, _output



# main
data_path = Path("/research/hutchinson/workspace/holmesa8/data/audio2text")
AudioDataset(data_path)
