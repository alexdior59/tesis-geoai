"""
Generadores de ortomosaicos sinteticos con la misma estructura que exporta Pix4Dfields para el
DJI Mavic 3M: 5 bandas float32 (Green, Red, Red edge, NIR, Alpha), nodata -10000 y EPSG:32618.
Las firmas espectrales tienen NDVI, NDRE y GNDVI conocidos para poder verificar los resultados.
"""
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

ORIGIN = (678000.0, 443000.0)
DESCRIPTIONS = ("Green", "Red", "Red edge", "NIR", "Alpha")

# Reflectancias (green, red, rededge, nir) por condicion
FIRMAS = {
    "vigorosa": (0.08, 0.03, 0.25, 0.55),     # NDVI 0.896
    "buena": (0.09, 0.05, 0.27, 0.45),        # NDVI 0.800
    "transicion": (0.10, 0.09, 0.20, 0.31),   # NDVI 0.550
    "estres": (0.10, 0.12, 0.22, 0.28),       # NDVI 0.400
    "suelo": (0.12, 0.18, 0.20, 0.21),        # NDVI 0.077
}


def ndvi_de(firma):
    _, red, _, nir = FIRMAS[firma]
    return (nir - red) / (nir + red)


def escribir_raster(path, bandas, res, descriptions=DESCRIPTIONS, nodata=-10000.0):
    """bandas: dict con green, red, rededge, nir (arreglos 2D) y alpha opcional."""
    h, w = bandas["nir"].shape
    alpha = bandas.get("alpha", np.full((h, w), 255, np.float32))
    stack = np.stack([bandas["green"], bandas["red"], bandas["rededge"], bandas["nir"], alpha]).astype(np.float32)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=5, dtype="float32",
                       crs="EPSG:32618", transform=from_origin(ORIGIN[0], ORIGIN[1], res, res),
                       nodata=nodata) as ds:
        ds.write(stack)
        if descriptions:
            for i, d in enumerate(descriptions, start=1):
                ds.set_band_description(i, d)
    return path


def mosaico_por_bloques(mapa, ruido=0.0, seed=0):
    """mapa: arreglo 2D de nombres de firma -> dict de bandas."""
    rng = np.random.default_rng(seed)
    bandas = {}
    for i, nombre in enumerate(("green", "red", "rededge", "nir")):
        valores = np.vectorize(lambda f: FIRMAS[f][i])(mapa).astype(np.float32)
        if ruido:
            valores = valores + rng.normal(0, ruido, valores.shape).astype(np.float32)
        bandas[nombre] = np.clip(valores, 0.001, 1.0)
    return bandas


@pytest.fixture
def lote_cuadrantes(tmp_path):
    """20 x 20 m a 10 cm: cuatro cuadrantes con NDVI 0.896, 0.55, 0.40 y 0.077 (25 % cada uno)."""
    mapa = np.empty((200, 200), dtype=object)
    mapa[:100, :100] = "vigorosa"
    mapa[:100, 100:] = "transicion"
    mapa[100:, :100] = "estres"
    mapa[100:, 100:] = "suelo"
    return escribir_raster(tmp_path / "cuadrantes.tif", mosaico_por_bloques(mapa), res=0.1)


def mapa_franjas():
    """Tres franjas horizontales de vegetacion (vigorosa, buena, estres): cada mitad del lote tiene las tres."""
    mapa = np.empty((240, 240), dtype=object)
    mapa[:80] = "vigorosa"
    mapa[80:160] = "buena"
    mapa[160:] = "estres"
    return mapa


@pytest.fixture
def lote_franjas(tmp_path):
    return escribir_raster(tmp_path / "franjas.tif", mosaico_por_bloques(mapa_franjas(), ruido=0.006), res=0.1)


@pytest.fixture
def lote_franjas_estresado(tmp_path):
    """Mismo lote, segundo vuelo simulado: el NIR de la mitad izquierda baja 15 %."""
    bandas = mosaico_por_bloques(mapa_franjas(), ruido=0.006)
    w = bandas["nir"].shape[1]
    bandas["nir"][:, : w // 2] *= 0.85
    return escribir_raster(tmp_path / "franjas_estres.tif", bandas, res=0.1)


SPACING = 9.0


def posiciones_malla(lado_m, spacing=SPACING):
    """Malla triangular (tres bolillo) dentro del lote, con margen de media distancia de siembra."""
    pos = []
    dy = spacing * np.sqrt(3) / 2
    j = 0
    while 4.5 + j * dy <= lado_m - 4.5:
        x0 = 4.5 + (spacing / 2 if j % 2 else 0.0)
        x = x0
        while x <= lado_m - 4.5:
            pos.append((x, 4.5 + j * dy))
            x += spacing
        j += 1
    return pos


def plantacion(lado_m=72.0, res=0.1, faltante=None, enferma=None):
    """
    Copas de palma como manchas gaussianas brillantes en NIR sobre cobertura de pasto.
    faltante / enferma: indice de la palma que se quita o que se pone palida y pequena.
    """
    n = int(round(lado_m / res))
    yy, xx = (np.mgrid[0:n, 0:n] + 0.5) * res
    pasto = np.array([0.09, 0.07, 0.20, 0.32], dtype=np.float32)
    copa = np.array([0.07, 0.03, 0.26, 0.60], dtype=np.float32)
    peso = np.zeros((n, n), dtype=np.float32)
    peso_enferma = np.zeros((n, n), dtype=np.float32)
    posiciones = posiciones_malla(lado_m)
    for k, (px, py) in enumerate(posiciones):
        if k == faltante:
            continue
        g = np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * 1.6 ** 2)).astype(np.float32)
        if k == enferma:
            peso_enferma = np.maximum(peso_enferma, g)
        else:
            peso = np.maximum(peso, g)
    bandas = {}
    palida = np.array([0.11, 0.11, 0.20, 0.27], dtype=np.float32)
    for i, nombre in enumerate(("green", "red", "rededge", "nir")):
        b = pasto[i] + peso * (copa[i] - pasto[i]) + 0.6 * peso_enferma * (palida[i] - pasto[i])
        bandas[nombre] = b.astype(np.float32)
    return bandas, posiciones


FALTANTE, ENFERMA = 34, 49  # palmas interiores de la malla (filas 4 y 6, centro del lote)


@pytest.fixture
def lote_palma(tmp_path):
    """Plantacion sintetica de 72 x 72 m (68 posiciones): una palma faltante y una enferma."""
    bandas, posiciones = plantacion(faltante=FALTANTE, enferma=ENFERMA)
    path = escribir_raster(tmp_path / "palma.tif", bandas, res=0.1)
    return path, posiciones
