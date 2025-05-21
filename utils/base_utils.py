import os

def read_file(ruta_archivo):
    with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
        contenido = archivo.read()
    return contenido

def load_prompts():
    prompts = {}
    prompts_dir_MY_PC = '/home/laura/ICA/Projetos 2024/miniKGraphRAG/prompts'
    prompts_dir = '/home/lauraparra85/ICA/miniKGraphRAG/prompts'
    for filename in os.listdir(prompts_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(prompts_dir, filename), 'r', encoding='utf-8') as file:
                prompts[filename] = file.read()
    return prompts

def load_dataset():
    datasets = {}
    dataset_dir = '/home/lauraparra85/ICA/miniKGraphRAG/KG data/'
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.json'):
            with open(os.path.join(dataset_dir, filename), 'r', encoding='utf-8') as file:
                datasets[filename] = json.load(file)
    return datasets