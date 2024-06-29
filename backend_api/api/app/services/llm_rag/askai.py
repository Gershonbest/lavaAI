from ....core.load_llms import groq_chat

def response(query: str):
    return groq_chat.invoke(query)