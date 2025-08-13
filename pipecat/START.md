# Voice Assistant Quickstart Guide

## Setup
0. Create a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
1. Install Pipecat:
```bash
pip install -r requirements.txt
```


2. Install Piper TTS:
Install epeak-ng if not already done
```bash
sudo port install espeak-ng
```

Check if espeak-ng is installed and works correctly:
```bash
# Find headers are installed correctly
find /usr/local/include -name speak_lib.h
find /opt/local/include -name speak_lib.h

# Test espeak-ng installation
espeak-ng --version
espeak-ng hello
```

If it is unable to find headers, you may need to set the `CFLAGS` environment variable:
```bash
export CFLAGS="-I/opt/local/include"
```

Download the voice models and config from [Piper repo](https://github.com/rhasspy/piper?tab=readme-ov-file#voices) and place them in the `models` directory.
```bash
mkdir -p models
```

Install Piper TTS HTTP Server
```bash
pip install piper-tts --no-deps piper-phonemize-cross onnxruntime numpy
```



## Usage

1. Start Piper TTS server:
```bash
.venv/bin/python3 -m piper.http_server --model ./models/en_US-lessac-medium.onnx --port 5151
```

2. Start Ollama API server:
```bash
ollama serve --model ollama/llama2
```

2. Start Pipecat Script:
```bash
python voice_ai.py
```



