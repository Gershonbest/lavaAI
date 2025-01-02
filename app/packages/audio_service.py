import time
import numpy as np
import pandas as pd
from app.models.models import WhisperModelHandler, DiarizationModelHandler
from app.packages.transcription.transcription import TranscriptionService
from app.packages.transcription.diarization import DiarizationService
from app.utils.audio_utils import process_audio, TranscriptionProcessor


class AudioProcessor(TranscriptionProcessor):
    def __init__(self, audio_url: str):
        self.audio_url = audio_url
        self.whisper_model = WhisperModelHandler()
        self.diarization_model = DiarizationModelHandler()
        self.diarization_service = DiarizationService(
            diarization_model=self.diarization_model
        )
        self.transcription_service = TranscriptionService(
            whisper_model=self.whisper_model
        )
        self.audio_data, self.audio_info = process_audio(self.audio_url)

    def run_diarization(self):

        diarization_result = self.diarization_service.diarize_audio(self.audio_info)
        return diarization_result

    def run_transcription(self):

        transcript, segments, language, duration = (
            self.transcription_service.transcribe_audio(self.audio_data)
        )


audio = AudioProcessor(audio_url="http")
