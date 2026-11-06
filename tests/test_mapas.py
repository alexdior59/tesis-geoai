import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from conftest import ORIGIN

RAIZ = Path(__file__).resolve().parents[1]

pytest.importorskip("matplotlib")


def correr(script, *args):
    r = subprocess.run([sys.executable, str(RAIZ / "experiments" / script), *map(str, args)],
                       capture_output=True, text=True, cwd=RAIZ)
    assert r.returncode == 0, r.stderr
    return r


def test_mapas_de_un_analisis_de_palma(lote_palma, tmp_path):
    path, _ = lote_palma
    salida = tmp_path / "resultado"
    correr("analisis_completo.py", path, salida, "--cultivo", "palma")
    correr("mapas.py", salida, "--tif", path, "--dpi", "60")

    esperados = ["01_zonas_fcm.png", "02_transicion.png", "03_incertidumbre.png",
                 "04_inspeccion_y_patrones.png", "05_palmas.png"]
    for nombre in esperados:
        png = salida / "mapas" / nombre
        assert png.exists(), nombre
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_mapas_con_el_poligono_del_lote(lote_palma, tmp_path):
    path, _ = lote_palma
    poligono = tmp_path / "lote.geojson"
    centro = box(ORIGIN[0] + 10, ORIGIN[1] - 62, ORIGIN[0] + 62, ORIGIN[1] - 10)
    gpd.GeoDataFrame(geometry=[centro], crs="EPSG:32618").to_file(poligono, driver="GeoJSON")

    salida = tmp_path / "resultado"
    correr("analisis_completo.py", path, salida, "--cultivo", "palma", "--lote", poligono)
    assert (salida / "lote.geojson").exists()
    correr("mapas.py", salida, "--dpi", "60")
    assert (salida / "mapas" / "01_zonas_fcm.png").exists()

    # sin --lote no debe quedar el limite de la corrida anterior
    correr("analisis_completo.py", path, salida, "--cultivo", "palma")
    assert not (salida / "lote.geojson").exists()
