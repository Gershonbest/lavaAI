from enum import Enum


class HuggingFace_Settings(Enum):
    LLAMA_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
    MIXTRAL_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
    MBART_50 = "https://api-inference.huggingface.co/models/facebook/mbart-large-50-many-to-many-mmt"