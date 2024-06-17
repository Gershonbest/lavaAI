from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain_groq import ChatGroq
import os

from .config import settings
api_key = settings.GROQ_API_KEY

# api_key = "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"

groq_chat = ChatGroq(
        groq_api_key=api_key,
        model_name="mixtral-8x7b-32768"
)
def load_groq_llm():
    chat = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", groq_api_key=api_key)

    system = "You are a helpful assistant."
    human = "{text}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    chain = prompt | chat
    print(chain.invoke({"text": "Explain the importance of low latency LLMs."}))