from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.controllers.analyze_router import router
from app.views.response_serializer import API_VERSION

app = FastAPI(
    title="Motor GeoAI",
    description=(
        "API REST para procesar ortomosaicos multiespectrales: condicion por rangos de NDVI por cultivo, "
        "zonas de inspeccion georreferenciadas, zonificacion Fuzzy C-Means + DBSCAN, conteo y analisis por "
        "palma, y comparacion contra una linea base validada por un experto. Los resultados no son "
        "diagnosticos agronomicos."
    ),
    version=API_VERSION,
    contact={
        "name": "Julian Diaz y Samuel Mejia",
        "email": "jadiaz@javeriana.edu.co",
    },
)

# TODO: restringir allow_origins cuando haya un frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", include_in_schema=False)
def root():
    return {
        "mensaje": "El motor GeoAI esta corriendo",
        "documentacion": "/docs",
        "health_check": "/api/v1/health",
    }