# Usamos una imagen oficial y ligera de Python
FROM python:3.12-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 1. Copiamos TODO el proyecto primero (incluyendo el código fuente y requirements)
COPY . /app

# 2. Instalamos las dependencias después de tener el código en el contenedor
RUN pip install --no-cache-dir -r requirements.txt

#  IMPORTANTE: Desinstalar pyopenssl (incompatible con cryptography)
RUN pip uninstall pyopenssl -y

# Exponer el puerto en el que corre FastAPI
EXPOSE 8000

# Indicamos al model_loader que estamos en el entorno Docker
ENV ENVIRONMENT=docker

# Comando para ejecutar la aplicación con Uvicorn
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]