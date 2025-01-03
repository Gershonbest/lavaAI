import time
import requests
from app.models.models import WhisperModelHandler, DiarizationModelHandler
from app.packages.speech_processing.transcription import TranscriptionService
from app.packages.speech_processing.diarization import DiarizationService
from app.utils.audio_utils import process_audio, TranscriptionProcessor

transcript_processing = TranscriptionProcessor()


class AudioProcessor:
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
        return self.diarization_service.diarize_audio(self.audio_info)

    def diarization_pipeline(self, diarization_df):
        text, segments, language, duration = (
            self.transcription_service.transcribe_audio(self.audio_data)
        )
        transcripts = self.transcript_processing.convert_faster_whisper_segments_to_openai_segment(
            segments
        )
        whisper_df = self.transcript_processing.segment_to_dataframe(transcripts)
        full_df = self.transcript_processing.combine_whisper_and_pyannote(
            text_df=whisper_df, speaker_df=diarization_df
        )
        combine_text = self.transcript_processing.combine_consecutive_speakers(full_df)
        diarized_output = self.transcript_processing.text_speaker_df_to_text(
            combine_text
        )
        return diarized_output, text, language, duration

    def translate_diarization(self, diarization_df, lang):
        text, segments, language, duration = (
            self.transcription_service.transcribe_audio(
                self.audio_data, task="translate", language=lang
            )
        )
        translated_text = self.transcript_processing.convert_faster_whisper_segments_to_openai_segment(
            segments
        )
        whisper_df = self.transcript_processing.segment_to_dataframe(translated_text)
        full_df = self.transcript_processing.combine_whisper_and_pyannote(
            text_df=whisper_df, speaker_df=diarization_df
        )
        combine_text = self.transcript_processing.combine_consecutive_speakers(full_df)
        diarized_output = self.transcript_processing.text_speaker_df_to_text(
            combine_text
        )
        return diarized_output, text

    def transcription_and_diarization(self):
        diarization_df = self.run_diarization()
        diarized_transcript, text, language, duration = self.diarization_pipeline(
            diarization_df
        )

        result = {
            "diarized_transcript": diarized_transcript,
            "text": text,
            "language": language.upper(),
            "duration": round(duration, 2),
        }

        if language != "en":
            diarized_translation, translation = self.translation_diarization_pipeline(
                diarization_df, language
            )
            result.update(
                {
                    "diarized_translation": diarized_translation,
                    "translation": translation,
                }
            )

        return result

    def process_transcription(self):
        # dispatcher_response_url = f"{response_url}/transcription/data"
        try:
            start_time = time.time()

            transcription_result = self.transcription_and_diarization()
            # transcription_result["extra_data"] = extra_data

            payload = {"data": transcription_result}
            end_time = time.time()
            elapsed_time = end_time - start_time

            print(f"Elapsed time: {elapsed_time} seconds")
            # print("Sending to dispatcher...", dispatcher_response_url, payload)
            # requests.post(url=dispatcher_response_url, json=payload)
            return payload

        except Exception as e:
            # error_payload = {
            #     "data": {"extra_data": {**extra_data, "model_error": str(e)}}
            # }
            # requests.post(url=dispatcher_response_url, json=error_payload)
            return False
