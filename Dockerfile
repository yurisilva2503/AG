FROM python:3.9-slim as builder

# 1. Instala todas as dependências necessárias
RUN apt-get update && \
    apt-get install -y nginx && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Copia e instala a API
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api/main.py .

# 3. Copia o frontend para o local correto do Nginx
COPY web/ /var/www/html/

# 4. Configuração do Nginx (com tratamento especial para Railway)
COPY nginx/nginx.conf /etc/nginx/nginx.conf

# 5. Script de inicialização
RUN echo "#!/bin/sh\n\
    sed -i \"s/listen .*;/listen \$PORT;/g\" /etc/nginx/nginx.conf\n\
    service nginx start\n\
    uvicorn main:app --host 0.0.0.0 --port 8000 &\n\
    tail -f /dev/null" > /start.sh && \
    chmod +x /start.sh

EXPOSE $PORT

CMD ["/start.sh"]