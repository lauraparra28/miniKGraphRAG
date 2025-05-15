from langchain_ollama import OllamaLLM

def load_llm(model_name="mistral"):
    return OllamaLLM(model=model_name)