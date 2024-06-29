import json, requests, os

from ....core.enums import HuggingFace_Settings
from ....core.config import settings
from ....utils.helpers import format_sentiment, remove_num_from_text

token = settings.HF_TOKEN_DEV
# token = os.environ.get("HF_TOKEN")

API_URL = HuggingFace_Settings.MIXTRAL_URL.value

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}


def text_summary(content):

    json_body = {
        "inputs": f"""[INST] <<SYS>> Give a well detailed summary of  the text.
        The content should contain only the summary and nothing else.<<SYS>> {content} [/INST] """,

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
    

def get_sentiment(content):

    sentiment_prompt = f"""
    [INST] <<SYS>> Your job is to extract sentiment from a text given to you. 
    Give your output in one word showing the sentiment in the conversation.
    Dont say any extra things after listing them.
    sentiments are positive, negative or neutral.<<SYS>> 
    
    {content} [/INST] 
    """
    json_body = {
        "inputs": sentiment_prompt,
        "parameters": {"max_new_tokens": 500, "top_p": 0.9, "temperature": 0.01},
    }
    data = json.dumps(json_body)

    try:
        response = requests.request("POST", API_URL, headers=headers, data=data)
        data = json.loads(response.content.decode("utf-8"))

        mood = data[0]["generated_text"].split("[/INST] ")[1]
        # mood = json.loads(mood)
        # mood = format_sentiment(mood)
        return mood

    except Exception:
        return {"Error": "Failed to get response for mood detection"}
    
def get_keywords(content):
    json_body = {
        "inputs": f"""[INST] <<SYS>> Your job is to extract special keywords from a text given to you. 
        Extract and list only 10 keywords which showcase the main topics. do not extract more then 10. 
        Dont say any extra things after listing them and the keywords must be in the content.
        example format: special, machine-learning, filter, logic. <<SYS>> {content} [/INST] """,
        "parameters": {"max_new_tokens": 256, "top_p": 0.9, "temperature": 0.01},
    }
    data = json.dumps(json_body)

    try:
        response = requests.request("POST", API_URL, headers=headers, data=data)
        data = json.loads(response.content.decode("utf-8"))
        tags = data[0]["generated_text"].split("[/INST] ")[1]
        tags = remove_num_from_text(tags)
        return tags
    except:
        return {"Error": "Failed to get response for tags !!!"}