
# 🧠 NeuroEvolução: Otimização de CNNs com Algoritmo Genético (AG)

Este projeto implementa uma **plataforma web interativa** para otimização de **redes neurais convolucionais (CNNs)** usando **Algoritmos Genéticos (AG)**, aplicados ao dataset CIFAR-10.

A aplicação foi desenvolvida com arquitetura modular conteinerizada em Docker, composta por:

- 🖥️ **Frontend** em HTML, CSS e JavaScript (com Bootstrap, Chart.js e Toastify)
- 🔧 **Backend** com API RESTful em FastAPI + PyTorch + NumPy + Scikit-Learn
- 🌐 **NGINX** como servidor estático e proxy reverso
- 🐳 Arquitetura de microsserviços via `docker-compose`

---

## 📁 Estrutura do Projeto

```
📦AG
 ┣ 📂api                 # Backend FastAPI + lógica do AG
 ┃ ┣ 📜main.py
 ┃ ┗ 📜requirements.txt
 ┣ 📂nginx               # Configuração do servidor NGINX
 ┃ ┗ 📜nginx.conf
 ┣ 📂web                 # Interface Web (frontend)
 ┃ ┣ 📂css
 ┃ ┣ 📂fonts
 ┃ ┣ 📂images
 ┃ ┣ 📂js
 ┃ ┗ 📜index.html
 ┣ 📜Dockerfile          # Dockerfile da aplicação
 ┗ 📜docker-compose.yml # Orquestração dos containers
```

---

## 🚀 Funcionalidades

- Configuração customizada de parâmetros do AG (número de gerações, taxa de mutação, etc.)
- Seleção múltipla de hiperparâmetros da CNN
- Visualização em tempo real da evolução do AG via gráficos
- Logs e progresso em tempo real
- Visualização de classificações corretas/incorretas
- Exportação de resultados (CSV, Excel, PDF)
- Armazenamento local de histórico de execuções

---

## 🧪 Tecnologias Utilizadas

### Backend:
- FastAPI
- Uvicorn
- PyTorch
- NumPy
- Scikit-learn

### Frontend:
- HTML5, CSS3, JavaScript
- Bootstrap 5
- Chart.js
- Toastify.js
- jsPDF
- DataTables

### Infraestrutura:
- Docker
- Docker Compose
- NGINX

---

## ⚙️ Como Executar Localmente

### Pré-requisitos

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)

### Passos

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/neuroevolucao.git
   cd neuroevolucao
   ```

2. Construa e suba os containers:
   ```bash
   docker-compose up --build
   ```

3. Acesse no navegador:
   ```
   http://localhost
   ```

---

## 📊 Demonstrações

### ⚙️ Configuração do AG
Permite definir tamanho da população, número de gerações, taxa de mutação, método de seleção (elitismo, roleta, torneio), etc.

### 📈 Visualização Gráfica
Gráficos dinâmicos mostram a acurácia média e o melhor indivíduo em cada geração.

### 🔁 Logs em tempo real
Acompanhe a execução geração por geração com mensagens diretas do backend.

### 🧠 Resultados e Classificações
Mostra as melhores redes e exemplos do CIFAR-10 corretamente e incorretamente classificados.

---

## 📦 Exportação de Resultados

- Exporta dados históricos para `.csv`, `.xlsx`, `.pdf`
- Inclui configurações do AG, logs, tempo de execução e acurácia

---

## 🛠️ Melhorias Futuras

- Persistência em banco de dados (PostgreSQL ou MongoDB)
- Execuções paralelas com Celery + Redis
- Dashboard analítico com análise de Pareto e heatmaps
- Suporte multiusuário com autenticação

---

## 👨‍💻 Autores

- Mateus Napoleão
- Warleson Sousa
- Yuri Gabriel

---

## 📜 Licença

Este projeto é acadêmico. Consulte sua instituição ou autores para uso comercial.
