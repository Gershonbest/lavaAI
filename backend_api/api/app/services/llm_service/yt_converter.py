from pytube import YouTube
import os

from ..audio_to_text.transcription import whisper_transcribe, detect_language


def load_youtube_audio(url):
 
    path = "./youtube_audio/"
    yt = YouTube(url)
    audio = yt.streams.filter(only_audio=True).first()

    audio.download(output_path=path)
    file_name = audio.default_filename
    source = path + file_name

    if ' ' in file_name:
        os.rename(source, source.replace(' ', '_'))
        file_name = source.replace(' ','_')

    file_without_ext = os.path.splitext(file_name)[0]
    audio_path = f"{file_without_ext}.mp3"
    command = f"ffmpeg -i {file_name} {file_without_ext}.mp3"

    os.system(command)
    os.remove(file_name)
    return audio_path


def youtube_to_text(url):
    try:
        audio_path = load_youtube_audio(url)
        language = detect_language(audio_path)
        result = whisper_transcribe(audio=audio_path, lang= language)

        try:
            os.remove(audio_path)
            print(f"File '{audio_path}' deleted successfully.")
        
        except FileNotFoundError:
            print(f"Error: File '{audio_path}' not found.")
            
        return result

    except KeyError:
        print("Unable to fetch video information. Please check the video URL or your network connection.")
        return None