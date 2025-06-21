FROM python:3.9-slim

# Instala Nginx e Supervisor
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

# Configuração completa do Supervisor
RUN echo "[supervisord]\n\
    nodaemon=true\n\
    logfile=/var/log/supervisor/supervisord.log\n\
    pidfile=/var/run/supervisord.pid\n\
    \n\
    [program:nginx]\n\
    command=nginx -g 'daemon off;'\n\
    autorestart=true\n\
    stdout_logfile=/dev/stdout\n\
    stdout_logfile_maxbytes=0\n\
    stderr_logfile=/dev/stderr\n\
    stderr_logfile_maxbytes=0\n\
    \n\
    [program:api]\n\
    command=uvicorn main:app --host 0.0.0.0 --port 8000\n\
    autorestart=true\n\
    directory=/app\n\
    stdout_logfile=/dev/stdout\n\
    stdout_logfile_maxbytes=0\n\
    stderr_logfile=/dev/stderr\n\
    stderr_logfile_maxbytes=0" > /etc/supervisor/conf.d/supervisord.conf

# Expõe a porta
EXPOSE $PORT

# Comando de inicialização
CMD ["sh", "-c", "sed -i \"s/listen .*;/listen $PORT;/g\" /etc/nginx/nginx.conf && \
    /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf"]