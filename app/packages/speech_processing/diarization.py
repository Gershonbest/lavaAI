import numpy as np
import pandas as pd
from app.models.models import DiarizationModelHandler
from app.schema.data_types import AudioInfo
from utils.logger import Logger

logger = Logger().get_logger()


class DiarizationService:
    def __init__(self,audio_file: AudioInfo, diarization_model: DiarizationModelHandler):
        self.diarization_model = diarization_model
        self.audio_file = audio_file
        
    def diarize_audio(self):
        try:
            # Get the number of speakers from the model
            num_speakers = self.diarization_model.num_speakers
            # Generate dynamic speaker mapping
            speaker_mapping = {
                f"SPEAKER_{str(i).zfill(2)}": f"SPEAKER {i + 1}" for i in range(num_speakers)
            }
            # Perform diarization
            diarization_result = self.diarization_model.diarize(self.audio_file)
            seg_info_list = []

            for speech_turn, track, speaker in diarization_result.itertrack(yield_label=True):
                speaker = speaker_mapping.get(speaker, speaker)  # Default to raw label if unmapped
                segment_info = {
                    "start": np.round(speech_turn.start, 2),
                    "end": np.round(speech_turn.end, 2),
                    "speaker": speaker,
                }
                segment_info_df = pd.DataFrame.from_dict({track: segment_info}, orient="index")
                seg_info_list.append(segment_info_df)

            logger.info(f"Diarization done for {num_speakers} speakers.")
            return pd.concat(seg_info_list, axis=0).reset_index()

        except Exception as e:
            logger.error(f"Error during diarization: {e}")
            raise