import os, io, tempfile

# import importlib.metadata
from fastapi import APIRouter, Form, File, UploadFile, Query
from fastapi.responses import StreamingResponse, RedirectResponse
from typing import BinaryIO, Union, Annotated


from ..services.llm_service.gen_summary import text_summary
from ...utils.helpers import process_audio
from ..services.asr_service.transcription import (
    whisper_transcribe,
    whisper_translate,
    detect_language,
)
from ...core.payloads import YouTubeDto, QuestionDto, TextObj
from ..services.llm_service.yt_converter import youtube_to_text
from ..services.llm_rag.askai import response
from ...core.payloads import source_languages


router = APIRouter()

@router.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"


@router.post("/speech-to-text", tags=["Endpoints"])
async def voice_to_text(
    file: UploadFile = File(...),
    task: Union[str, None] = Query(
        default="transcribe", enum=["transcribe", "translate"]
    ),
    lang: Union[str, None] = Query(default=None, description="The language code for translation (optional)"),
):
    """
    This Endpoint processes an uploaded audio file and performs either transcription or translation based on the task parameter.

    Parameters:
    file (UploadFile): The uploaded audio file. Must be in one of the following formats: 'audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/flac'.
    task (str): The task to be performed on the audio file. Must be either 'transcribe' or 'translate'.

    Returns:
    dict: A dictionary containing the language of the audio file and the result of the task.
    If the task is 'transcribe', the dictionary will contain the transcribed text. If the task is 'translate',
    the dictionary will contain the translated text. If the file format is invalid or the task is invalid, an error message will be returned.
    """

    if file.content_type not in ["audio/mpeg", "audio/wav", "audio/ogg", "audio/flac"]:
        return {"error": "Invalid file format. Please upload an audio file."}

    audio = process_audio(file.file)
    # language = detect_language(audio) if lang is None else language = lang
    if lang is None:
        language = detect_language(audio)
        print(f"Detected language: {language}")
    else: language = lang
    
    if task == "transcribe":
        result = whisper_transcribe(audio=audio, lang=language)
        returned_result = {"language": language, "transcription": result}
        return returned_result
    elif task == "translate":
        result = whisper_translate(audio=audio, lang=language)
        returned_result = {"language": language, "translation": result}
        return returned_result
    else:
        return {"error": "Invalid task. Please choose either transcribe or translate."}


@router.post("/transcribe-youtube", tags=["Endpoints"])
def transcribe_youtube(youtube: YouTubeDto):
    if youtube_to_text is None:
        return {"error": "Invalid YouTube URL. Please enter a valid YouTube URL."}
    result = youtube_to_text(youtube.video_url)
    summary = text_summary(result)

    return {"transcription": result, "summary": summary}

@router.post("/ai-summary", tags=["Endpoints"])
def ai_summary(text: TextObj):
    """
    This endpoint takes in a text input and returns a an AI generated summary.

    Parameters:
        text (str): The text input to be summarized.

    Returns:
        (dict): A dictionary containing the text and summary.
    """
    summary = text_summary(text.input_text)
    print(summary)
    return {"text": text, "summary": summary}

@router.post("/emotion-from-text", tags=["Endpoints"])
def emotion_from_text(text: TextObj):
    pass


@router.post("/sentiment-from-text", tags=["Endpoints"])
def sentiment_from_text(text: TextObj):
    pass


@router.post("/keyword-from-text", tags=["Endpoints"])
def keywords_from_text(text: TextObj):
    pass


@router.post("/ask-ai", tags=["Endpoints"])
def ask_ai(query: QuestionDto):
    result = response(query.question)
    return {"question": query.question, "answer": result.content}
