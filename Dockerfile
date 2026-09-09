FROM python:3.11-slim

# Instalamos GDAL a nivel de sistema operativo porque si no rasterio no funciona en Linux
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    gcc \
    g++

WORKDIR /geoai

# Copiamos esto primero para que docker no vuelva a descargar todas las librerias cada vez que cambiamos un .py
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copiamos todo el codigo fuente de la app
COPY app/ ./app/

# Carpeta temporal para que FastAPI guarde los archivos pesados mientras procesa
RUN mkdir -p /tmp/geoai_uploads

EXPOSE 8000

# Comando para prender el servidor web
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]