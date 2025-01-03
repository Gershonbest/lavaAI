import torch
from typing import Union
from faster_whisper.vad import VadOptions
from faster_whisper.transcribe import TranscriptionOptions
from app.models.models import WhisperModelHandler
from utils.logger import Logger

logger = Logger().get_logger()


class TranscriptionService:
    def __init__(self, whisper_model: WhisperModelHandler):
        self.whisper_model = whisper_model
        self.vad_options = VadOptions(
            threshold=0.25,
            min_speech_duration_ms=50,
            min_silence_duration_ms=500,
            speech_pad_ms=1000,
        )

    def transcribe_audio(
        self,
        audio_file: Union[torch.Tensor, bytes, str],
        language: str = None,
        task: str = "transcribe",
    ):
        options_dict = {
            "task": task,
            "word_timestamps": True,
            "beam_size": 1,
            "vad_filter": True,
            "vad_parameters": self.vad_options,
        }
        if language:
            options_dict["language"] = language

        segment_generator, transcription_info = self.whisper_model.transcribe(
            audio_file, **options_dict
        )
        segments = []
        text = ""
        for segment in segment_generator:
            segments.append(segment)
            text += segment.text

        torch.cuda.empty_cache()
        logger.info("Audio transcription done.")
        return (
            text,
            segments,
            transcription_info.language,
            transcription_info.duration,
        )
