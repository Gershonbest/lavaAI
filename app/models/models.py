import torch
from typing import Union, Iterable, Tuple, NamedTuple, TypedDict
from faster_whisper import WhisperModel
from faster_whisper.transcribe import TranscriptionInfo, Segment
from pyannote.audio import Pipeline
from schema.data_types import AudioInfo

from config import config


class WhisperModelHandler:
    def __init__(self):
        self.model_size = config.WHISPER_MODEL_SIZE
        self.model = WhisperModel(
            self.model_size, device=config.DEVICE, compute_type=config.COMPUTER_TYPE
        )
 
    def transcribe(
        self, audio_file: Union[torch.Tensor, bytes, str], **options: Union[dict, None]
    ) -> Tuple[ Iterable[Segment], TranscriptionInfo]:
        """Transcribes the audio file into text segments"""
        return self.model.transcribe(audio_file, **options)
    

class DiarizationModelHandler:
    def __init__(self, number_of_speakers: int = 2):

        """Initializes the diarization model"""
        self.num_speakers = number_of_speakers
        self.model_name = config.DIARIZATION_MODEL_NAME
        self.hf_token = config.HF_TOKEN
        self.pipeline = Pipeline.from_pretrained(self.model_name, use_auth_token=self.hf_token)
        if config.DEVICE == "cuda":
            self.pipeline.to(torch.device(config.DEVICE))
        
    def diarize(
        self, audio_data: Union[AudioInfo, dict]
    ) -> Iterable[Tuple[float, float, str]]:
        
        """Diarizes the audio file into segments"""
        return self.pipeline(audio_data, num_speakers= self.num_speakers)
