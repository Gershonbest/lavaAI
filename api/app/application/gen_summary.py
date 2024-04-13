import json, requests, os

from ...core.enums import HuggingFace_Settings
from ...core.config import settings

token = settings.HF_TOKEN_DEV
# token = os.environ.get("HF_TOKEN")

API_URL = HuggingFace_Settings.MIXTRAL_URL.value

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}


def text_summary(payload):

    json_body = {
        "inputs": f"""[INST] <<SYS>> Give a well detailed summary of  the text.
        The content should contain only the summary and nothing else.<<SYS>> {payload} [/INST] """,

        "parameters": {"max_new_tokens": 256, "top_p": 0.9, "temperature": 0.01},
    }
    data = json.dumps(json_body)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    try:
        data = json.loads(response.content.decode("utf-8"))
        summary = data[0]["generated_text"].split("[/INST] ")[1]
        summary = summary.replace("\n", "")
        return summary
    except Exception:
        return {"Error": "Failed to get response from inference api for summary!!!"}