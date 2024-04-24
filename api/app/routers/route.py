import os, io, tempfile
from fastapi import APIRouter, Form, File, UploadFile
from ..services.llm_summary.gen_summary import text_summary
from ...utils.helpers import process_audio
from ..services.audio_to_text.transcription import whisper_transcribe, whisper_translate, detect_language
from ...core.payloads import YouTubeDto
from ..services.llm_summary.yt_converter import youtube_to_text

router = APIRouter()

@router.get("/")
def home():
    return {"Hello": "World"}


@router.post("/ai-summary")
def ai_summary(text: str = Form(...)):
    """
    This function takes in a text input and returns a summary of the text using the text_summary function.

    Parameters:
        text (str): The text input to be summarized.

    Returns:
        (dict): A dictionary containing the summary of the text.
    """
    summary = text_summary(text)
    print(summary)
    return {"text":text, "summary": summary}

@router.post("/voice-to-text")
async def voice_to_text(file: UploadFile = File(...), task= str):
    
    if file.content_type not in ['audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/flac']:
        return {"error": "Invalid file format. Please upload an audio file."}

    audio = process_audio(file.file)
    language = detect_language(audio)
    if task == "transcribe":
        result = whisper_transcribe(audio=audio, lang= language)
        returned_result = {
        "language": language,
        "transcription": result
    }
        return returned_result
    elif task == "translate":
        result = whisper_translate(audio=audio, lang= language)
        returned_result = {
        "language": language,
        "translation": result
    }
        return returned_result
    else:
        return {"error": "Invalid task. Please choose either transcribe or translate."}
    
    

@router.post("/transcribe-youtube")
def transcribe_youtube(youtube:YouTubeDto):
    if youtube_to_text is None:
        return {"error": "Invalid YouTube URL. Please enter a valid YouTube URL."}
    result = youtube_to_text(youtube.video_url)
    summary = text_summary(result)
    
    return {
        "transcription": result,
        "summary": summary
        }
