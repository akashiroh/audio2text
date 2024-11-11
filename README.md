# Audio2Text

This repo contains a simple implementation of audio-to-text transcription using pre-trained models

## Setup

**Conda**

- First, download [Conda](https://docs.anaconda.com/free/miniconda/) or check that it is already installed
- Clone the repo:

```
git clone git@github.com:akashiroh/audio2text.git
cd audio2text
```

```
conda create -n ENVIRONMENT_NAME
conda install pytorch::torchaudio # do this first for highest chance of success
conda install -c conda-forge ffmpeg # needed as a backend for torchaudio to process .wav files
```

## Dataset

This repo works with audio files (.wav) and transcriptions of what was said in those videos

- [LibriSpeech ASR Corpus](http://www.openslr.org/12) (.flac)
- [VoxForge](http://www.repository.voxforge1.org/downloads/en/Trunk/Audio/Main/16kHz_16bit/) (.wav)

## Models

- Loaded pre-trained models from:
	- Speechbrain
- Tokenized targets with built-in tokenizer from speechbrain
- Evaluate model using jiwer word error rate
