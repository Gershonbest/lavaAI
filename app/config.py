import os
import torch

class Config:
    APP_NAME = "AI Model Serving"
    HOST = "0.0.0.0"
    PORT = 8080
    HF_TOKEN = os.getenv("HF_TOKEN", "your_huggingface_token")
    EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2"
    WHISPER_MODEL_SIZE = "medium"
    DIARIZATION_MODEL_NAME = "pyannote/speaker-diarization-3.1"
    if torch.cuda.is_available():
        DEVICE = "cuda"
        COMPUTER_TYPE = "float16"
    else:
        DEVICE = "cpu"
        COMPUTER_TYPE = "int8"




config = Config()