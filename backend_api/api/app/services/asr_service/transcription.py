import torch, whisper
import timeit
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
import numpy as np
from typing import Union, List, Dict, Tuple

from ....core.config import settings
from ....core.payloads import source_languages

os.environ["HF_HOME"] = "/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/.cache/"

# token = os.environ.get("HF_TOKEN")
token = settings.HF_TOKEN_DEV

torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-medium"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=False, use_safetensors=True
)

processor = AutoProcessor.from_pretrained(model_id)
whisper_model = whisper.load_model('medium', device= device)


def audio_pipeline(language: str, task: str = 'transcribe') -> pipeline:
    """
    Whisper audio pipeline. For translating or transcribing any giving audio in different language
    to English.

    Args:
        language (str): language code of the audio file
        task (str, optional): task to be performed. Defaults to 'transcribe'.

    Returns:
        pipeline: a Whisper pipeline for automatic speech recognition, transcription and translation
    """
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=32,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs={"language": language, "task": f"{task}"},
    )
    return pipe


def whisper_transcribe(lang: str, audio: np.ndarray) -> str:
    """Performs speech to text translation.

    Args:
        lang (str): language code of the audio file
        audio (np.ndarray, audio path): audio data

    Returns:
        str: Transcribed text
    """
    audio_to_text = audio_pipeline(language=lang)
    result = audio_to_text(audio)
    torch.cuda.empty_cache()
    return result["text"]


def whisper_translate(lang: str, audio: np.ndarray) -> str:
    """Uses the whisper text translation pipeline to translate a given text to English"""

    audio_to_text = audio_pipeline(language=lang, task= "translate")
    result = audio_to_text(audio)
    torch.cuda.empty_cache()
    return result["text"]

def detect_language(audiofile):
    """ Detects the language of a given audio file using openai-whisper"""

    audio = whisper.load_audio(audiofile)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
    _, probs = whisper_model.detect_language(mel)
    language = max(probs, key=probs.get)
    print("Language: {language}".format(language=language))
    return language


