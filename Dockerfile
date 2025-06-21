FROM python:3.9-slim

# Instala Nginx e supervidor (para gerenciar múltiplos processos)
RUN apt-get update && \
    apt-get install -y nginx supervisor && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia a API
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api/main.py .

# Copia o frontend
COPY web/ /var/www/html/

# Configuração do Nginx
COPY nginx/nginx.conf /etc/nginx/nginx.conf

# Configuração do Supervisor
RUN echo "[program:nginx]\n\
    command=nginx -g 'daemon off;'\n\
    autorestart=true\n\
    \n\
    [program:api]\n\
    command=uvicorn main:app --host 0.0.0.0 --port 8000\n\
    autorestart=true\n\
    directory=/app" > /etc/supervisor/conf.d/supervisord.conf

# Expõe a porta (Railway usará a variável PORT)
EXPOSE $PORT

# Comando de inicialização
CMD ["sh", "-c", "sed -i \"s/listen .*;/listen $PORT;/g\" /etc/nginx/nginx.conf && \
    /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf"]