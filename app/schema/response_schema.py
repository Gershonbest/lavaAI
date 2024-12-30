from pydantic import BaseModel
from typing import Optional

class TranscriptionOutputBody(BaseModel):
    text: str = ""
    diarized_transcript: str = ""
    translation: str = ""
    diarized_translation: str = ""
    duration: float = None
    language: str = None
    extra_data: Optional[dict] = None

    