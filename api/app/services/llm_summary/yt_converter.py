from pytube import YouTube
import os, logging

from ..audio_to_text.transcription import whisper_transcribe, detect_language


def youtube_to_audio(url):

    try:
        video = YouTube(url)
        stream = video.streams.filter(only_audio=True).first()
        out_file = stream.download(filename=f"{video.title}.mp3")
        file_stats = os.stat(out_file)

        logging.info(f"Downloaded {file_stats.st_size / 1024 / 1024} MB")
        base, ext = os.path.splitext(out_file)
        new_file = base+'.mp3'
        os.rename(out_file, new_file)
        return new_file

    except KeyError:
        print("Unable to fetch video information. Please check the video URL or your network connection.")
        return None

def youtube_to_text(url):
    try:
        video = YouTube(url)
        stream = video.streams.filter(only_audio=True).first()
        out_file = stream.download(filename=f"{video.title}.mp3")
        file_stats = os.stat(out_file)

        logging.info(f"Downloaded {file_stats.st_size / 1024 / 1024} MB")
        base, ext = os.path.splitext(out_file)
        new_file = base+'.mp3'
        os.rename(out_file, new_file)
        
        language = detect_language(new_file)
        result = whisper_transcribe(audio=new_file, lang= language)

        return result


    except KeyError:
        print("Unable to fetch video information. Please check the video URL or your network connection.")
        return None