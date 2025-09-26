import requests
print(requests.get("http://192.168.128.1:11500/api/tags").json())

print("✅ Conexión exitosa a Ollama ")
