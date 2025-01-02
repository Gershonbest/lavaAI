import requests
import tempfile
import os
import io

import time
import torchaudio
import logging
import pandas as pd
import numpy as np
from fastapi import HTTPException
from typing import Tuple
from faster_whisper.
from app.schema.data_types import AudioInfo
from app.utils.logger import Logger

logger = Logger().get_logger()


def get_audio_data(audio_url: str) -> str:
    try:
        logger.info("Downloading Audio...")
        # audio_bytes = requests.get(audio_url).content
        response= requests.post(url="http://18.194.85.55:3001/voip-call-record/bite-data", json={
            "fileUrl": audio_url
            }
        ).json()
        try:
            audio_data = response['data']
            audio_bytes = bytes(audio_data)
        except KeyError:
            raise HTTPException(status_code=400, detail="'data' key not found in response")
        

    except requests.exceptions.ConnectionError:
        logger.error("connection error in downloading error")
        return {"error":"connection error"}
    except requests.exceptions.Timeout:
        logger.error("request timeout in downloading error")
        return {"error":"The request timed out."}
    except requests.exceptions.HTTPError:
        logger.error("HTTP Error from audio_url")
        return {"error":"HTTP Error from audio_url"}    
    except requests.exceptions.RequestException:
        logger.error("An error occurred, check audio url")
        return {"error":"An error occurred, check audio url"}

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            with open(temp_file.name, "wb") as f:
                f.write(audio_bytes)
                if os.path.exists(temp_file.name):
                    with open(temp_file.name, "rb") as f:
                        file_reader = io.BufferedReader(f)
                        audio_data = temp_file.name
                        temp_file.close()
    
    return audio_data


def process_audio(url)-> Tuple[str, AudioInfo]:
    
    audio_data = get_audio_data(audio_url= url)

    waveform, sample_rate = torchaudio.load(audio_data)
    logger.info("Audio loaded")
    audio_info = AudioInfo(waveform, sample_rate)
    return audio_data, audio_info



class TranscriptionProcessor:
    """
    A class to process audio transcription and diarization results.
    """
    @staticmethod
    def convert_faster_whisper_segments_to_openai_segment(segments:):
        openai_segments = []
        for segment in segments:
            id,_, start, end, text, _,_,_,_,_,_ = segment

            openai_segments.append({"id":id-1, "start":start, "end":end, "text":text})

        return openai_segments
    
    @staticmethod
    def combine_whisper_and_pyannote(text_df, speaker_df):

        text_df = text_df.loc[:, ['id','start', 'end', 'text']]
        speaker_df = speaker_df.loc[:, ['index', 'start', 'end', 'speaker']]

        overlap_list = []
        for idx, this_row in speaker_df.iterrows():
            this_start = this_row['start']
            this_end = this_row['end']
            this_speaker = this_row['speaker']
            
            
            xx_inds =~ ((text_df['end'] < this_start) | (text_df['start'] > this_end))
            this_overlap_texts = text_df.loc[xx_inds, :]

            this_overlap_texts['speaker_start'] = this_start
            this_overlap_texts['speaker_end'] = this_end
            this_overlap_texts['speaker'] = this_speaker

            overlap_list.append(this_overlap_texts)
            
        all_overlaps = pd.concat(overlap_list)
        all_overlaps = all_overlaps.reset_index(drop=True)

        all_overlaps['max_start'] = np.maximum(all_overlaps['start'], 
                                            all_overlaps['speaker_start'])

        all_overlaps['min_end'] = np.minimum(all_overlaps['end'], 
                                            all_overlaps['speaker_end'])

        all_overlaps['overlap_duration'] = all_overlaps['min_end'] - all_overlaps['max_start']

        # pick only one text/speaker combination for each text

        max_overlap_indices = all_overlaps.groupby('id')['overlap_duration'].idxmax()
        text_speaker_df = all_overlaps.loc[max_overlap_indices, :]

        return text_speaker_df
    
    @staticmethod
    def convert_whisper_output(whisper_output, audio_duration):
        """Converts Whisper output to a dictionary of segments with start and end timestamps."""

        segments = []
        for i, item in enumerate(whisper_output):

            start = item['timestamp'][0] 
            end = audio_duration if item['timestamp'][1] is None and i == len(whisper_output) - 1 else item['timestamp'][1]
            text = item['text']

            segments.append({'id': i, 'start': start, 'end': end, 'text': text})
            
        print(f"the last value for i is :{i}")
        return segments
    @staticmethod
    def segment_to_dataframe(transcription_result):
        """
        Converts the segments into a pandas DataFrame with columns for
        segment start, end, and text.
        """
        df = pd.DataFrame(transcription_result, columns=["id", "start", "end", "text"])
        df["start"] = df["start"].apply(lambda x: round(x, 3))
        df["end"] = df["end"].apply(lambda x: round(x, 3))
        return df

    @staticmethod
    def combine_consecutive_speakers(text_speaker_df_raw):
        """
        Combines consecutive segments by the same speaker.
        """
        text_speaker_df = text_speaker_df_raw.copy()
        n_iter = text_speaker_df.shape[0]

        for counter in range(1, n_iter):
            is_same_speaker = (
                text_speaker_df['speaker'].iloc[counter] == text_speaker_df['speaker'].iloc[counter - 1]
            )
            if is_same_speaker:
                new_start = text_speaker_df['start'].iloc[counter - 1]
                previous_text = text_speaker_df['text'].iloc[counter - 1]
                new_text = previous_text + ' ' + text_speaker_df['text'].iloc[counter]

                text_speaker_df.at[counter, 'start'] = new_start
                text_speaker_df.at[counter, 'text'] = new_text
                text_speaker_df.at[counter - 1, 'start'] = np.nan
                text_speaker_df.at[counter - 1, 'end'] = np.nan

        text_speaker_df = text_speaker_df.dropna().loc[:, ['start', 'end', 'text', 'speaker']]
        text_speaker_df = text_speaker_df.reset_index(drop=True)
        text_speaker_df = text_speaker_df.sort_values('start')

        return text_speaker_df

    @staticmethod
    def text_speaker_df_to_text(text_speaker_df):
        """
        Converts a text-speaker DataFrame into a formatted string.
        """
        output_str = ''
        for _, this_row in text_speaker_df.iterrows():
            this_speaker = this_row['speaker']
            this_text = this_row['text']
            output_str += f'{this_speaker}: {this_text}\n'
        return output_str

    @staticmethod
    def insert_timestamp(segments):
        """
        Inserts timestamps into the transcript text.
        """
        transcript = ""
        for segment in segments:
            transcript += (
                "\n"
                + "["
                + time.strftime("%H:%M:%S", time.gmtime(segment["start"]))
                + "] "
            )
            transcript += segment["text"][1:] + " "
        return transcript