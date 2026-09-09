from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.controllers.analyze_router import router

app = FastAPI(
    title="Motor GeoAI",
    description="API REST para procesar ortomosaicos multiespectrales con K-Means y DBSCAN. Ojo: los resultados no son diagnosticos agronomicos.",
    version="1.0.0",
    contact={
        "name": "Julian Diaz y Samuel Mejia",
        "email": "jdiaz@javeriana.edu.co",
    }
)

# TODO: Cuando haya un frontend o alguien vaya a consumir la API, cambiar el allow_origins para que no quede publico
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", include_in_schema=False)
def root():
    # Ruta por defecto para que no salga Not Found si entran a la raiz
    return {
        "mensaje": "El motor GeoAI esta corriendo",
        "documentacion": "/docs",
        "health_check": "/api/v1/health",
    }