from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def load_llm_with_api_key(model_name="gpt-4o-mini"):
    return ChatOpenAI(
        model_name=model_name,
        temperature=0.0,
        max_tokens=128,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )   

def load_llm(model_name="llama3:8b"): # MI PC: deepseek-r1:1.5b  qwen2.5:1.5b gemma3:1b
    return OllamaLLM(
        model=model_name,
        temperature=0.0,
        max_tokens=128,
        base_url="http://127.0.0.1:11434"
    )
    
def load_cypher_llm(model_cypher="llama3:8b"): # MI PC: qwen2.5:1.5b funcionó para generación CYPHER
    return OllamaLLM(
        model=model_cypher, # Probar en el ICA con "ollama run codellama:7b # MUY PESADO"
        temperature=0.0
    )