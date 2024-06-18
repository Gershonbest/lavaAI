import tempfile, os, io


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