# Singing Voice Conversion

A simple singing voice conversion demo and inference workflow.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Model

Download the model from the following link:

[5_HP-Karaoke-UVR.pth](https://huggingface.co/HitPawOfficial/HitPawVoice/resolve/main/SingingVoiceConversion/5_HP-Karaoke-UVR.pth)

Place the downloaded file in the following directory:

```text
lib/vocal_remover/5_HP-Karaoke-UVR.pth
```

## Usage

Main entry file:

```text
demo_svc.py
```

## Main Parameters

- `sid`: speaker ID, `int`, range `[0, 18]`
- `f0_up_key`: pitch shift, `int`, range `[-12, 12]`  
  - Use `0` for male-to-male or female-to-female conversion  
  - Use `+12` for male-to-female conversion  
  - Use `-12` for female-to-male conversion
- `index_rate`: voice similarity, `float`, default `0.7`, range `[0, 1]`

## Requirements

See `requirements.txt` for the full environment dependencies.