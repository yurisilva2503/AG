
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import uuid
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import uvicorn
# Configuração inicial do FastAPI
app = FastAPI(title="API de Algoritmo Genético para CIFAR-10",
             description="API para otimização de hiperparâmetros de CNN usando Algoritmo Genético")

# Configuração de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ou especifique ["http://127.0.0.1:5500"]
    allow_credentials=True,
    allow_methods=["*"],  # Ou especifique ["GET", "POST", "OPTIONS"]
    allow_headers=["*"],
    expose_headers=["*"]  # Adicione esta linha
)

# ========================
# 1. Modelos Pydantic para a API
# ========================
class AGConfig(BaseModel):
    pop_size: int = 5
    geracoes: int = 5
    taxa_mutacao: float = 0.4
    metodo_selecao: str = "torneio"  # "elitismo", "torneio" ou "roleta"
    elite_size: int = 2
    tamanho_torneio: int = 3
    n_epochs: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    space: Dict[str, List[Any]] = Field(default_factory=dict)
    
    # Configuração adicional para o modelo (opcional)
    model_config = ConfigDict(extra='forbid')  # Isso previne campos extras

class Individuo(BaseModel):
    params: Dict[str, Any]
    accuracy: float

class AGResponse(BaseModel):
    task_id: str
    status: str
    progress: int
    best_individual: Optional[Individuo]
    history: Optional[List[List[float]]]
    message: Optional[str]

class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: int
    result: Optional[Dict[str, Any]]

# ========================
# 2. Configurações Globais e Estado
# ========================
default_space = {
    "learning_rate": [1e-3, 5e-4, 1e-4, 3e-4],
    "batch_size": [16, 32, 64],
    "n_filters": [16, 64],
    "n_fc": [32, 64, 128, 256, 512],
    "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
    "weight_decay": [0, 1e-4, 5e-4, 1e-3, 3e-4],
    "activation": ["relu", "leaky_relu", "elu"],
    "optimizer": ["adamw", "sgd"],
    "stride": [1],
    "aggregation": ["sum", "concat", "max", "avg"]
}

# Estado das tarefas em execução
tasks = {}

# Lista de logs global
task_logs = {}  # {task_id: [log1, log2, ...]}

# Pool de threads para execução assíncrona
executor = ThreadPoolExecutor(max_workers=4)

# Carregar dados uma vez
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

full_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_idx, val_idx = train_test_split(range(len(full_trainset)), test_size=0.2, random_state=42)
trainset = Subset(full_trainset, train_idx[:30000])
valset = Subset(full_trainset, val_idx[:10000])
full_valset = Subset(full_trainset, val_idx)

# ========================
# 3. Modelo CNN
# ========================
class ImprovedCNN(nn.Module):
    def __init__(self, n_filters, n_fc, dropout, activation, stride, aggregation):
        super().__init__()
        self.activation_name = activation
        self.aggregation = aggregation

        self.block1 = nn.Sequential(
            nn.Conv2d(3, n_filters, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(n_filters, n_filters * 2, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(n_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(n_filters * 2, n_filters * 4, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(n_filters * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        if aggregation in ["sum", "max", "avg"]:
            self.adjust2 = nn.Conv2d(n_filters * 2, n_filters, kernel_size=1)
            self.adjust3 = nn.Conv2d(n_filters * 4, n_filters, kernel_size=1)

        self.dropout = nn.Dropout(dropout)
        self.n_fc = n_fc
        self.flatten_dim = None
        self.fc1 = None
        self.fc2 = None

    def build(self, device):
        with torch.no_grad():
            x = torch.zeros(1, 3, 32, 32).to(device)
            out1 = self.block1(x)
            out2 = self.block2(out1)
            out3 = self.block3(out2)

            if self.aggregation == "sum":
                out2 = self.adjust2(out2)
                out3 = self.adjust3(out3)
                x = out1 + F.interpolate(out2, size=out1.shape[2:]) + F.interpolate(out3, size=out1.shape[2:])
            elif self.aggregation == "concat":
                x = torch.cat([
                    F.interpolate(out1, size=out3.shape[2:]),
                    F.interpolate(out2, size=out3.shape[2:]),
                    out3
                ], dim=1)
            elif self.aggregation == "max":
                out2 = self.adjust2(out2)
                out3 = self.adjust3(out3)
                x = torch.maximum(
                    out1,
                    torch.maximum(
                        F.interpolate(out2, size=out1.shape[2:]),
                        F.interpolate(out3, size=out1.shape[2:])
                    )
                )
            elif self.aggregation == "avg":
                out2 = self.adjust2(out2)
                out3 = self.adjust3(out3)
                x = (out1 + F.interpolate(out2, size=out1.shape[2:]) + F.interpolate(out3, size=out1.shape[2:])) / 3
            else:
                raise ValueError(f"Aggregation não suportada: {self.aggregation}")

            self.flatten_dim = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flatten_dim, self.n_fc).to(device)
        self.fc2 = nn.Linear(self.n_fc, 100).to(device)

    def activate(self, x):
        if self.activation_name == "relu":
            return F.relu(x)
        elif self.activation_name == "leaky_relu":
            return F.leaky_relu(x)
        elif self.activation_name == "elu":
            return F.elu(x)
        else:
            raise ValueError(f"Ativação não suportada: {self.activation_name}")

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)

        if self.aggregation == "sum":
            out2 = self.adjust2(out2)
            out3 = self.adjust3(out3)
            x = out1 + F.interpolate(out2, size=out1.shape[2:]) + F.interpolate(out3, size=out1.shape[2:])
        elif self.aggregation == "concat":
            x = torch.cat([
                F.interpolate(out1, size=out3.shape[2:]),
                F.interpolate(out2, size=out3.shape[2:]),
                out3
            ], dim=1)
        elif self.aggregation == "max":
            out2 = self.adjust2(out2)
            out3 = self.adjust3(out3)
            x = torch.maximum(
                out1,
                torch.maximum(
                    F.interpolate(out2, size=out1.shape[2:]),
                    F.interpolate(out3, size=out1.shape[2:])
                )
            )
        elif self.aggregation == "avg":
            out2 = self.adjust2(out2)
            out3 = self.adjust3(out3)
            x = (out1 + F.interpolate(out2, size=out1.shape[2:]) + F.interpolate(out3, size=out1.shape[2:])) / 3
        else:
            raise ValueError(f"Aggregation não suportada: {self.aggregation}")

        x = x.view(x.size(0), -1)

        if self.fc1 is None or self.fc2 is None:
            raise ValueError("Chame o método build(device) após instanciar o modelo!")

        x = self.dropout(self.activate(self.fc1(x)))
        x = self.fc2(x)
        return x

# ========================
# 4. Funções do Algoritmo Genético
# ========================
def criar_individuo(space):
    individuo = {}
    
    # Garante que todos os parâmetros necessários estejam presentes
    required_params = [
        "learning_rate", "batch_size", "n_filters", "n_fc", 
        "dropout", "weight_decay", "activation", 
        "optimizer", "stride", "aggregation"
    ]
    
    for param in required_params:
        if param in space and space[param]:  # Verifica se existe e não está vazio
            individuo[param] = random.choice(space[param])
        else:
            individuo[param] = default_space.get(param, [None])[0]
    
    # Validação adicional para n_fc >= n_filters
    if "n_fc" in individuo and "n_filters" in individuo:
        while individuo["n_fc"] < individuo["n_filters"] and random.random() > 0.2:
            if "n_fc" in space and space["n_fc"]:
                individuo["n_fc"] = random.choice(space["n_fc"])
            else:
                individuo["n_fc"] = individuo["n_filters"]  # Garante pelo menos o mesmo tamanho
    
    return individuo

def crossover(pai1, pai2):
    filho = {}
    for key in pai1:
        filho[key] = random.choice([pai1[key], pai2[key]])
    return filho

def mutar(individuo, space):
    # Seleciona uma chave que existe tanto no indivíduo quanto no espaço
    valid_keys = [k for k in individuo.keys() if k in space]
    if not valid_keys:
        return individuo
        
    chave = random.choice(valid_keys)
    individuo[chave] = random.choice(space[chave])
    return individuo

def avaliar_fitness(individuo, device, n_epochs=1, save_preds=False):
    train_loader = DataLoader(trainset, batch_size=individuo["batch_size"],
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=individuo["batch_size"],
                            shuffle=False, num_workers=0, pin_memory=True)
    model = ImprovedCNN(
        n_filters=individuo["n_filters"],
        n_fc=individuo["n_fc"],
        dropout=individuo["dropout"],
        activation=individuo["activation"],
        stride=individuo["stride"],
        aggregation=individuo["aggregation"]
    )

    model = model.to(device)
    model.build(device)

    criterion = nn.CrossEntropyLoss()

    if individuo["optimizer"] == "adamw":
        optimizer = optim.AdamW(model.parameters(),
                                lr=individuo["learning_rate"],
                                weight_decay=individuo.get("weight_decay", 0))
    elif individuo["optimizer"] == "sgd":
        optimizer = optim.SGD(model.parameters(),
                              lr=individuo["learning_rate"],
                              momentum=0.9,
                              weight_decay=individuo.get("weight_decay", 0))
    else:
        raise ValueError(f"Otimização não suportada: {individuo['optimizer']}")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    model.train()
    for epoch in range(n_epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            _, predicted = torch.max(pred, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
            total += yb.size(0)
            correct += (predicted == yb).sum().item()
    acc = correct / total

    if save_preds:
        # Seleciona exemplos corretos e incorretos
        correct_samples = []
        wrong_samples = []
        
        # Pegamos até 4 exemplos de cada tipo
        for idx in range(len(all_labels)):
            if len(correct_samples) < 6 and all_preds[idx] == all_labels[idx]:
                correct_samples.append(idx)
            elif len(wrong_samples) < 6 and all_preds[idx] != all_labels[idx]:
                wrong_samples.append(idx)
            if len(correct_samples) >= 6 and len(wrong_samples) >= 6:
                break
        
        sample_images = []
        
        def process_image(idx):
            img, _ = valset[idx]
            # Converte para numpy e desfaz a normalização
            img = img.numpy()
            img = np.transpose(img, (1, 2, 0))  # (C, H, W) -> (H, W, C)
            mean = np.array([0.5071, 0.4867, 0.4408])
            std = np.array([0.2675, 0.2565, 0.2761])
            img = img * std + mean  # Desfaz a normalização
            img = np.clip(img, 0, 1)  # Garante valores entre 0 e 1
            return img.tolist()
        
        for idx in correct_samples:
            sample_images.append({
                'image': process_image(idx),
                'pred': int(all_preds[idx]),
                'true': int(all_labels[idx]),
                'correct': True
            })
        
        for idx in wrong_samples:
            sample_images.append({
                'image': process_image(idx),
                'pred': int(all_preds[idx]),
                'true': int(all_labels[idx]),
                'correct': False
            })
        
        return acc, np.array(all_preds), np.array(all_labels), sample_images
    return acc

def selecao_elitismo(populacao, fitness, elite_size):
    if not populacao:
        return []
        
    elite_size = min(elite_size, len(populacao))
    melhores = sorted(zip(populacao, fitness), key=lambda x: x[1], reverse=True)
    return [ind for ind, _ in melhores[:elite_size]]

def selecao_torneio(populacao, fitness, num_selecionados, tamanho_torneio=3):
    selecionados = []
    if len(populacao) == 0:
        return selecionados
        
    # Garante que o tamanho do torneio não seja maior que a população
    tamanho_torneio = min(tamanho_torneio, len(populacao))
    
    for _ in range(num_selecionados):
        competidores = random.sample(list(zip(populacao, fitness)), tamanho_torneio)
        vencedor = max(competidores, key=lambda x: x[1])[0]
        selecionados.append(vencedor)
    return selecionados

def selecao_roleta(populacao, fitness, num_selecionados):
    if not populacao:
        return []
        
    fitness = np.array(fitness)
    fitness = np.clip(fitness, 0, None)  # Remove valores negativos
    
    # Se todos os fitness forem zero, usa seleção uniforme
    if np.sum(fitness) == 0:
        return random.sample(populacao, min(num_selecionados, len(populacao)))
    
    prob = fitness / np.sum(fitness)
    
    # Garante que não tentamos selecionar mais do que a população
    num_selecionados = min(num_selecionados, len(populacao))
    indices = np.random.choice(len(populacao), size=num_selecionados, p=prob, replace=True)
    return [populacao[i] for i in indices]

def run_ag(task_id, config: AGConfig):
    task_logs[task_id] = []

    def log(msg):
        print(msg)
        task_logs[task_id].append(msg)
        
    try:
        tasks[task_id]["status"] = "running"
        tasks[task_id]["start_time"] = time.time()

        # Extrai todos os parâmetros da configuração
        pop_size = config.pop_size
        geracoes = config.geracoes
        taxa_mutacao = config.taxa_mutacao
        device = config.device
        metodo_selecao = config.metodo_selecao
        elite_size = config.elite_size
        tamanho_torneio = config.tamanho_torneio
        space = config.space
        n_epochs = config.n_epochs

        historico = []
        melhor_ind = None
        best_acc = 0

        # Adiciona log inicial
        log(f"Iniciando AG com {pop_size} indivíduo(s) e {geracoes} geração(oes)")
        log(f"Espaço de busca: {space}")  # Mostra o espaço de busca recebido
        log(f"-" * 20)

        # Inicialização
        populacao = []
        for _ in range(pop_size):
            ind = criar_individuo(space)
            while ind["n_fc"] < ind["n_filters"] and random.random() > 0.2:
                ind = criar_individuo(space)
            populacao.append(ind)

        for g in range(geracoes):
            log_msg = f"Geração {g+1}/{geracoes}"
            log(log_msg)
            tasks[task_id]["progress"] = int((g / geracoes) * 100)

            fitness = []
            for i, ind in enumerate(populacao):
                start_time = time.time()  # Inicia o cronômetro
                acc = avaliar_fitness(ind, device, n_epochs)
                elapsed_time = time.time() - start_time  # Calcula tempo decorrido
                
                # Formata os parâmetros para exibição
                params_str = {
                    'learning_rate': ind["learning_rate"],
                    'batch_size': ind["batch_size"],
                    'n_filters': ind["n_filters"],
                    'n_fc': ind["n_fc"],
                    'dropout': ind["dropout"],
                    'weight_decay': ind["weight_decay"],
                    'activation': ind["activation"],
                    'optimizer': ind["optimizer"],
                    'stride': ind["stride"],
                    'aggregation': ind["aggregation"]
                }
                
                # Log detalhado do indivíduo
                log_entry = (
                    f"Indivíduo {i+1}: Acurácia: {acc:.4f} | "
                    f"Parâmetros: {params_str} | "
                    f"Tempo: {elapsed_time:.1f}s"
                )
                log(log_entry)
                log(f" " * 10)
                
                fitness.append(acc)

            # Registra histórico
            melhores = sorted(zip(populacao, fitness), key=lambda x: x[1], reverse=True)
            historico.append([fit for _, fit in melhores[:elite_size]])

            # Atualiza o melhor indivíduo global
            current_best_acc = max(fitness)
            if current_best_acc > best_acc:
                best_acc = current_best_acc
                melhor_ind = populacao[fitness.index(best_acc)]
                tasks[task_id]["best_individual"] = {
                    "params": melhor_ind,
                    "accuracy": best_acc
                }
                # Log do melhor indivíduo da geração
                log(f"Melhor da geração {g+1}: Acurácia: {best_acc:.4f}")

            if not populacao:
                log("População vazia! Abortando...")
                tasks[task_id]["status"] = "failed"
                tasks[task_id]["error"] = "População vazia"
                return
            
            # Processo de seleção (mantido igual)
            if metodo_selecao == 'elitismo':
                elite_size = min(config.elite_size, len(populacao))
                nova_populacao = selecao_elitismo(populacao, fitness, elite_size)
                selecionados = [ind for ind, _ in melhores[:pop_size//2]]
            elif metodo_selecao == 'torneio':
                tamanho_torneio = min(config.tamanho_torneio, len(populacao))
                selecionados = selecao_torneio(populacao, fitness, pop_size//2, tamanho_torneio)
                nova_populacao = []
            elif metodo_selecao == 'roleta':
                selecionados = selecao_roleta(populacao, fitness, pop_size//2)
                nova_populacao = []
            else:
                raise ValueError(f"Método de seleção desconhecido: {metodo_selecao}")
                
            # Garante que temos selecionados suficientes
            if len(selecionados) < 2:
                selecionados = random.sample(populacao, min(2, len(populacao)))

            # Preenche a nova população com crossover e mutação
            while len(nova_populacao) < pop_size:
                pai1, pai2 = random.sample(selecionados, 2)
                filho = crossover(pai1, pai2)
                if random.random() < taxa_mutacao:
                    filho = mutar(filho, space)
                nova_populacao.append(filho)

            populacao = nova_populacao

        # Resultado final
        log("\nAG finalizado com sucesso.")
        log(f"Melhor acurácia encontrada: {best_acc:.4f}")
        log(f"Melhor indivíduo: {melhor_ind}")
        
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["end_time"] = time.time()
        tasks[task_id]["history"] = historico
        tasks[task_id]["execution_time"] = tasks[task_id]["end_time"] - tasks[task_id]["start_time"]

        # Avaliar o melhor indivíduo final com predições
        final_acc, preds, labels, sample_images = avaliar_fitness(melhor_ind, device, n_epochs, save_preds=True)
        tasks[task_id]["predictions"] = {
            "preds": preds.tolist(),
            "labels": labels.tolist(),
            "sample_images": sample_images  # Adiciona as imagens de exemplo
        }

    except Exception as e:
        log(f"Erro na tarefa {task_id}: {str(e)}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        raise e

# ========================
# 5. Endpoints da API
# =======================

@app.post("/start_ag", response_model=AGResponse, tags=["Algoritmo Genético"])
async def start_ag(config: AGConfig):
    task_id = str(uuid.uuid4())
    
    # Processa o espaço de busca para manter apenas os valores selecionados
    filtered_space = {}
    if config.space:
        for key, values in config.space.items():
            if values:  # Só inclui se valores foram fornecidos
                filtered_space[key] = values
    
    # Usa o espaço filtrado ou o default se nenhum valor foi selecionado para algum parâmetro
    final_space = default_space.copy()
    for key in filtered_space:
        final_space[key] = filtered_space[key]
    
    tasks[task_id] = {
        "status": "pending",
        "progress": 0,
        "config": {
            "pop_size": config.pop_size,
            "geracoes": config.geracoes,
            "taxa_mutacao": config.taxa_mutacao,
            "metodo_selecao": config.metodo_selecao,
            "elite_size": config.elite_size,
            "tamanho_torneio": config.tamanho_torneio,
            "n_epochs": config.n_epochs,
            "device": config.device,
            "space": final_space  # Usa o espaço com valores selecionados
        },
        "start_time": None,
        "end_time": None,
        "best_individual": None,
        "history": None,
        "execution_time": None,
        "predictions": None
    }

    # Executa em background, sem bloquear
    def wrapper():
        try:
            run_ag(task_id, AGConfig(**tasks[task_id]["config"]))
        except Exception as e:
            print(f"[ERRO] Task {task_id} falhou: {e}")

    executor.submit(wrapper)

    return {
        "task_id": task_id,
        "status": "pending",
        "progress": 0,
        "best_individual": None,
        "history": None,
        "message": "O algoritmo genético foi iniciado. Use o endpoint /status para verificar o progresso."
    }

@app.get("/status/{task_id}", response_model=TaskStatus, tags=["Algoritmo Genético"])
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task ID not found")

    task = tasks[task_id]
    response = {
        "task_id": task_id,
        "status": task["status"],
        "progress": task["progress"],
        "result": None
    }

    if task["status"] == "completed":
        response["result"] = {
            "best_individual": task["best_individual"],
            "history": task["history"],
            "execution_time": task["execution_time"],
            "config": task["config"],
            "predictions": task["predictions"]
        }
    elif task["status"] == "failed":
        response["result"] = {
            "error": task.get("error", "Unknown error")
        }

    return response

@app.post("/stop_task/{task_id}", tags=["Algoritmo Genético"])
async def stop_task(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task ID not found")
    
    if tasks[task_id]["status"] in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Task already finished")
    
    tasks[task_id]["status"] = "stopped"
    tasks[task_id]["end_time"] = time.time()
    tasks[task_id]["execution_time"] = tasks[task_id]["end_time"] - tasks[task_id]["start_time"]
    
    return {"status": "success", "message": f"Task {task_id} stopped"}

@app.get("/list_tasks", response_model=Dict[str, str], tags=["Algoritmo Genético"])
async def list_tasks():
    return {task_id: task["status"] for task_id, task in tasks.items()}

@app.get("/search_space", tags=["Configuração"])
async def get_search_space():
    return default_space

@app.get("/sample_images/{task_id}")
async def get_sample_images(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task ID not found")
    
    task = tasks[task_id]
    
    # Verifica se a tarefa foi concluída
    if task.get("status") != "completed":
        raise HTTPException(
            status_code=400, 
            detail="Task not completed yet"
        )
    
    # Verifica se existem predições
    if not task.get("predictions"):
        raise HTTPException(
            status_code=404,
            detail="No predictions available"
        )
    
    # Verifica se existem imagens de exemplo
    if "sample_images" not in task["predictions"]:
        raise HTTPException(
            status_code=404,
            detail="Sample images not generated"
        )
    
    return task["predictions"]["sample_images"]


@app.get("/quick_run", response_model=AGResponse, tags=["Exemplo Rápido"])
async def quick_run():
    """Endpoint para executar uma configuração rápida do AG"""
    config = AGConfig(
        pop_size=3,
        geracoes=2,
        taxa_mutacao=0.3,
        metodo_selecao="elitismo",
        space=default_space,
        n_epochs=1  # Adicionado número de épocas
    )
    return await start_ag(config)

@app.get("/logs/{task_id}")
async def get_task_logs(task_id: str):
    if task_id not in task_logs:
        raise HTTPException(status_code=404, detail="Task não encontrada")
    return {"logs": task_logs[task_id]}

@app.get("/health", tags=["Monitoramento"])
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

# Execução (normalmente você usaria uvicorn na linha de comando)
if __name__ == "__main__":

    # authtoken = "2ibwiuhnvnMY65gmCe8uRxAjLy1_6xHUNq4GcuLJBQ6akpsZR"  #authtoken do ngrok
    # ngrok.set_auth_token(authtoken)

    # Expor o servidor local para a internet
    # public_url = ngrok.connect(8000)
    # print(f"Public URL: {public_url}")

    # Rodar o servidor uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
    print("Servidor rodando em http://localhost:8000")