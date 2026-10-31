"""
Exactitud tematica del mapa de zonas (Congalton, 1991; Olofsson et al., 2014).

1. formulario: a partir de la carpeta generada por analisis_completo.py, crea el Excel a ciegas
   para el experto (un punto por fila, con coordenadas y enlace al mapa, sin la zona del modelo).
2. evaluar: con el Excel diligenciado, la clave de zonas y el nombre que el experto le dio a cada
   zona, calcula la matriz de confusion, la exactitud global, del usuario, del productor y Kappa.

Uso:
    python experiments/exactitud_tematica.py formulario <carpeta_resultados>
    python experiments/exactitud_tematica.py evaluar <excel_diligenciado> <clave_puntos_zona.csv> --etiquetas etiquetas.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.models.accuracy import accuracy_assessment  # noqa: E402

CATEGORIAS = [
    "Vegetación sana", "Estrés leve", "Estrés severo", "Planta muerta o faltante", "Suelo desnudo o camino",
    "Maleza o arvenses", "Pasto u otro cultivo", "Bosque o árboles", "Agua o encharcamiento",
    "Senescencia o madurez (normal)", "Sombra o artefacto", "No se puede determinar",
]
COLUMNAS = ["punto_id", "lat", "lon", "mapa", "categoria_observada", "confianza", "observaciones"]


def formulario(carpeta):
    carpeta = Path(carpeta)
    gj = json.loads((carpeta / "puntos_verificacion.geojson").read_text(encoding="utf-8"))
    wb = Workbook()
    ws = wb.active
    ws.title = "Puntos"
    ws["A1"] = ("Verificacion a ciegas: para cada punto, indique que hay en el terreno o en la imagen. "
                "Solo se llenan las celdas amarillas.")
    ws["A1"].font = Font(italic=True)
    for j, h in enumerate(COLUMNAS, start=1):
        c = ws.cell(row=3, column=j, value=h)
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="2F5D3A")
    amarillo = PatternFill("solid", fgColor="FFF2B3")
    feats = sorted(gj["features"], key=lambda f: f["properties"]["punto_id"])
    for i, f in enumerate(feats, start=4):
        pr = f["properties"]
        ws.cell(row=i, column=1, value=pr["punto_id"])
        ws.cell(row=i, column=2, value=pr["lat"])
        ws.cell(row=i, column=3, value=pr["lon"])
        link = ws.cell(row=i, column=4, value="Abrir")
        link.hyperlink = f"https://www.google.com/maps?q={pr['lat']},{pr['lon']}"
        link.font = Font(color="1155CC", underline="single")
        for j in (5, 6, 7):
            ws.cell(row=i, column=j).fill = amarillo
    ultimo = 3 + len(feats)
    dv = DataValidation(type="list", formula1='"' + ",".join(CATEGORIAS) + '"', allow_blank=True)
    dv2 = DataValidation(type="list", formula1='"Alta,Media,Baja"', allow_blank=True)
    ws.add_data_validation(dv)
    ws.add_data_validation(dv2)
    dv.add(f"E4:E{ultimo}")
    dv2.add(f"F4:F{ultimo}")
    for col, w in zip("ABCDEFG", (9, 12, 13, 8, 30, 11, 45)):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "B4"
    salida = carpeta / "verificacion_exactitud.xlsx"
    wb.save(salida)
    print(f"Formulario con {len(feats)} puntos: {salida}")


def leer_texto(path):
    datos = Path(path).read_bytes()
    try:
        return datos.decode("utf-8-sig")
    except UnicodeDecodeError:
        return datos.decode("cp1252")


def evaluar(excel, clave, etiquetas):
    zonas = {}
    with open(clave, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            zonas[str(row["punto_id"])] = str(row["zona"])
    nombres = json.loads(leer_texto(etiquetas)) if etiquetas else {}
    ws = load_workbook(excel, data_only=True)["Puntos"]
    mapa, referencia = [], []
    for r in ws.iter_rows(min_row=4, values_only=True):
        pid, obs = r[0], r[4]
        if pid is None or not obs:
            continue
        z = zonas[str(pid)]
        mapa.append(nombres.get(z, f"zona {z}"))
        referencia.append(obs)
    res = accuracy_assessment(mapa, referencia)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    Path(excel).with_name("exactitud_tematica.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                                                  encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("formulario")
    f.add_argument("carpeta")
    e = sub.add_parser("evaluar")
    e.add_argument("excel")
    e.add_argument("clave")
    e.add_argument("--etiquetas", help='JSON con el nombre del experto para cada zona: {"2": "Palma vigorosa"}')
    a = ap.parse_args()
    if a.cmd == "formulario":
        formulario(a.carpeta)
    else:
        evaluar(a.excel, a.clave, a.etiquetas)


if __name__ == "__main__":
    main()
