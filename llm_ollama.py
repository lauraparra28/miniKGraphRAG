from langchain_ollama import OllamaLLM

def load_llm(model_name="gemma3:1b"): # MI PC: deepseek-r1:1.5b  qwen2.5:1.5b gemma3:1b
    return OllamaLLM(
        model=model_name,
        temperature=0.1,
        max_tokens=128,
        base_url="http://127.0.0.1:11434" # cat /etc/resolv.conf | grep nameserver 10.255.255.254 192.168.134.59
    )
    
def load_cypher_llm(model_cypher="qwen2.5:1.5b"): # MI PC: qwen2.5:1.5b funcionó para generación CYPHER
    return OllamaLLM(
        model=model_cypher, # Probar en el ICA con "ollama run codellama:7b"
        temperature=0.0
    )