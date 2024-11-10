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

        transcription_file = data_path / "1272-135031.trans.txt"

        tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        collect = []
        self.tokenized= []
        with open(transcription_file, "r") as f:
            for line in f.readlines():
                file_name = line[:16]
                transcription = line[17:].strip()
                tokenized_transcription = tokenizer.encode(transcription)
                self.tokenized.append(tokenized_transcription)
                collect.append(
                    (file_name, transcription)
                )
        print("Collected transcriptions")

        self.df = pd.DataFrame(
            collect,
            columns=["file_name", "transcription"]
        )

        collect = []
        for file in self.df["file_name"]:
            file = data_path / f"{file}.flac"
            breakpoint()
            waveform, sr = torchaudio.load(file)
            if sr != sample_rate:
                waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)

            if transform:
                waveform = transform(waveform)

            collect.append(waveform)
        print("Processed Audio Files")

        # self.waveforms = torch.concat(collect, dim=?)

        print(f"Audio Dataset | returned {len_self.df} datapoints")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        transcription = self.tokenized[idx]
        waveform = self.waveforms[idx]

        return _input, _output



# main
data_path = Path("/home/holmesa8/repos/audio_data/135031")
AudioDataset(data_path)
