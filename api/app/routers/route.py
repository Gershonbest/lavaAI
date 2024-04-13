import os, io, tempfile
from fastapi import APIRouter, Form, File, UploadFile
from ..application.gen_summary import text_summary

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
async def voice_to_text(file: UploadFile = File(...)):
    
    if file.content_type not in ['audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/flac']:
        return {"error": "Invalid file format. Please upload an audio file."}
    
    try:
        # Load the audio data securely
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            audio_bytes = await file.read()

            # Copy the raw file data to the temporary file
            with open(temp_file.name, 'wb') as f:
                f.write(audio_bytes)
                if os.path.exists(temp_file.name):
           
                    with open(temp_file.name, 'rb') as f:
                        file_reader = io.BufferedReader(f)
                        print(file_reader)
                        audio = temp_file.name
                        temp_file.close()

                        # return voice_to_text_service(audio)

    except Exception as e:
        return {"error": str(e)}  