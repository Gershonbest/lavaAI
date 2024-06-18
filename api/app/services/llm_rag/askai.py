from ....core.load_llms import groq_chat

def response(query):
    return groq_chat.invoke(query)