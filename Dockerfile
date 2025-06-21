# Dockerfile único
FROM python:3.9-slim

# Instala Nginx e dependências
RUN apt-get update && \
    apt-get install -y nginx && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia e instala a API
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api/main.py .

# Copia os arquivos estáticos do frontend
COPY web/ /var/www/html/

# Copia a configuração do Nginx
COPY nginx/nginx.conf /etc/nginx/conf.d/default.conf

# Expõe a porta (Railway usará a variável PORT)
EXPOSE $PORT

# Comando para iniciar ambos serviços
CMD service nginx start && uvicorn main:app --host 0.0.0.0 --port 8000