import torch
import requests
import time
import json
import numpy as np
from typing import Union, Tuple
from pyannote.audio import Pipeline
import pandas as pd
from helpers import (
    logger,
    combine_consecutive_speakers,
    combine_whisper_and_pyannote,
    segment_to_dataframe,
    text_speaker_df_to_text,
    convert_faster_whisper_segments_to_openai_segment,
    process_audio,
)

from data_body import TranscriptionOutputBody
from models import whisper_models
from config import settings
from faster_whisper.vad import VadOptions

class AudioProcessor:
    def __init__(self, hf_token: str, num_speakers: int = 2):
        self.hf_token = hf_token
        self.num_speakers = num_speakers
        
        self.vad_options = VadOptions(
            threshold=0.25,
            min_speech_duration_ms=50,
            min_silence_duration_ms=500,
            speech_pad_ms=1000,
        )

    def get_transcripts(self, model, audio_file: Union[torch.Tensor, bytes, str], task: str = "transcribe", language: str = None) -> Tuple[str, Tuple, str, str]:
        logger.info("Start generating the transcription...")
        torch.cuda.empty_cache()
        options_dict = {
            "task": task,
            "word_timestamps": True,
            "beam_size": 1,
            "vad_filter": True,
            "vad_parameters": self.vad_options,
        }
        if language:
            options_dict["language"] = language

        segment_generator, info = model.transcribe(audio_file, **options_dict)

        segments = []
        text = ""
        for segment in segment_generator:
            segments.append(segment)
            text += segment.text

        result = {
            "language": info.language,
            "duration": info.duration,
            "segments": segments,
            "text": text,
        }
        torch.cuda.empty_cache()
        logger.info("Audio transcription done.")
        return result["text"], result["segments"], result["language"], result["duration"]

    def get_diarization(self, diarization_model: str, waveform_sample_rate):
        logger.info("Start generating the diarization...")
        torch.cuda.empty_cache()
        pipeline = Pipeline.from_pretrained(diarization_model, use_auth_token=self.hf_token)
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        diarization_result = pipeline(waveform_sample_rate, num_speakers=self.num_speakers)
        torch.cuda.empty_cache()

        seg_info_list = []
        for speech_turn, track, speaker in diarization_result.itertracks(yield_label=True):
            speaker = "SPEAKER 1" if speaker == "SPEAKER_00" else "SPEAKER 2"
            segment_info = {
                "start": np.round(speech_turn.start, 2),
                "end": np.round(speech_turn.end, 2),
                "speaker": speaker,
            }
            segment_info_df = pd.DataFrame.from_dict({track: segment_info}, orient="index")
            seg_info_list.append(segment_info_df)
        torch.cuda.empty_cache()
        logger.info("Diarization done.")
        return pd.concat(seg_info_list, axis=0).reset_index()

    def diarization_pipeline(self, audio_file, model, diarization_df):
        text, segments, language, duration = self.get_transcripts(model=model, audio_file=audio_file)
        transcripts = convert_faster_whisper_segments_to_openai_segment(segments=segments)
        whisper_df = segment_to_dataframe(transcripts)
        full_df = combine_whisper_and_pyannote(text_df=whisper_df, speaker_df=diarization_df)
        combine_text = combine_consecutive_speakers(full_df)
        diarized_output = text_speaker_df_to_text(combine_text)

        return diarized_output, text, language, duration

    def translation_diarization_pipeline(self, audio_file, model, diarization_df, lang):
        text, segments, language, duration = self.get_transcripts(model=model, audio_file=audio_file, task="translate", language=lang)
        translated_text = convert_faster_whisper_segments_to_openai_segment(segments=segments)
        whisper_df = segment_to_dataframe(translated_text)
        full_df = combine_whisper_and_pyannote(text_df=whisper_df, speaker_df=diarization_df)
        combine_text = combine_consecutive_speakers(full_df)
        diarized_output = text_speaker_df_to_text(combine_text)

        return diarized_output, text

    def transcription_and_diarization(self, audio_str_path, audio_raw_data, whisper_model, transcription_data_body):
        diarization_df = self.get_diarization(
            diarization_model="pyannote/speaker-diarization-3.1",
            waveform_sample_rate=audio_raw_data,
        )

        (
            transcription_data_body.diarized_transcript,
            transcription_data_body.text,
            language,
            duration,
        ) = self.diarization_pipeline(
            audio_file=audio_str_path,
            model=whisper_model,
            diarization_df=diarization_df,
        )
        transcription_data_body.duration = round(duration, 2)
        if language != "en":
            (
                transcription_data_body.diarized_translation,
                transcription_data_body.translation,
            ) = self.translation_diarization_pipeline(
                audio_file=audio_str_path,
                model=whisper_model,
                diarization_df=diarization_df,
                lang=language,
            )

        transcription_data_body.language = language.upper()
        logger.info("Completed!")
        return transcription_data_body

    def process_audio_request(self, audio_url, extra_data, response_url):
        transcription_data = TranscriptionOutputBody()
        transcription_data.extra_data = extra_data
        dispatcher_response_url = f"{response_url}/transcribtion/data"
        try:
            start_time = time.time()
            model = whisper_models()
            logger.info("Whisper Model loaded!")
            audio_data, waveform_sample_rate = process_audio(audio_url)

            output = self.transcription_and_diarization(
                audio_str_path=audio_data,
                audio_raw_data=waveform_sample_rate,
                whisper_model=model,
                transcription_data_body=transcription_data,
            ).json()
            output = json.loads(output)
            payload = {"data": output}
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f"Elapsed time: {elapsed_time} seconds")
            print("sending to dispatcher......", dispatcher_response_url, payload)
            requests.post(url=dispatcher_response_url, json=payload)
            return True

        except Exception as e:
            logger.info(f"Error: {e}")
            transcription_data = TranscriptionOutputBody().dict()
            transcription_data["extra_data"] = extra_data
            transcription_data["extra_data"]["model_error"] = str(e)
            payload = {"data": transcription_data}
            requests.post(url=dispatcher_response_url, json=payload)
            return True