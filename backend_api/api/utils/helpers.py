import tempfile, os, io, re


def process_audio(audio_file):
    
    try:
        # Load the audio data securely
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            audio_bytes = audio_file.read()

            # Copy the raw file data to the temporary file
            with open(temp_file.name, 'wb') as f:
                f.write(audio_bytes)
                if os.path.exists(temp_file.name):
           
                    with open(temp_file.name, 'rb') as f:
                        file_reader = io.BufferedReader(f)
                        print(file_reader)
                        audio = temp_file.name
                        temp_file.close()
                        return audio
    except Exception as e:
        print(e)
        return {"error": "Error processing audio file."}
    

def remove_num_from_text(text: str) -> list:
    """
    This function takes a text as input and returns a list of words without numbers and punctuation.
    Parameters:
    -----------
    text : str
        The text to be processed
    Returns:
    --------
    list
        A list of words without numbers and punctuation
    """
    no_numbers_punctuation = re.sub(r"[\d!#.]", "", text)
    words = no_numbers_punctuation.splitlines()
    return words

def format_sentiment(text: str) -> dict:
    """
    This function takes a sentiment analysis text as input and returns a dictionary of emotions.
    Parameters:
    -----------
    text : str
        The sentiment analysis text
    Returns:
    --------
    dict
        A dictionary of emotions with their corresponding percentages
    """
    text = text.strip().split(",")
    emotions = {}

    for emotion_text in text:
        emotion, percentage = emotion_text.strip().split("(")
        percentage = percentage[:-1]
        emotions[emotion.strip()] = percentage

    for emotion_content, value in emotions.items():
        emotions[emotion_content] = re.sub(r"[)]", "", value)
    return emotions