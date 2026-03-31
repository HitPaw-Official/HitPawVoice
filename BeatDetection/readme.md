# Beat Detection

A simple beat detection demo and inference workflow.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
cd madmom
python setup.py develop
```

## Model

Download the model from the following link:

[model_1_weights.onnx](https://huggingface.co/HitPawOfficial/HitPawVoice/resolve/main/BeatDetection/model_1_weights.onnx)

Place the downloaded file in the `models` directory:

```text
models/model_1_weights.onnx
```

## Usage

Run the demo with:

```bash
python demo.py --input data/test/1_3beats.mp3 --model models/model_1_weights.onnx
```

## Project Structure

```text
.
├── data
│   ├── test
│   └── outputs
├── models
│   └── model_1_weights.onnx
├── demo.py
└── README.md
```

## Notes

- Test audio files are stored in `data/test`
- Reference outputs are stored in `data/outputs`