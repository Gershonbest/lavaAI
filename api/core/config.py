import os
from dotenv import load_dotenv
from pathlib import Path

dir_path = (Path(__file__) / ".." / ".." / "..").resolve()
env_path = os.path.join(dir_path, ".env")

load_dotenv(dotenv_path=env_path)


class Settings:
    
    HF_TOKEN_DEV= os.getenv('HF_TOKEN')
    

settings = Settings()