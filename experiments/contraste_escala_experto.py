"""
Contraste entre la reflectancia de la imagen y la escala de NDVI del codirector (Acta N. 01).

Aplica la escala (> 0.6 buena condicion, 0.5-0.6 transicion, 0.2-0.5 estres, < 0.2 suelo o
senescencia) a cada punto de verificacion y, si se dan, a cada poligono grande, y la compara con el
rango que el modelo le asigno a cada zona. Solo mide la consistencia del mapa con la regla de
umbrales y no reemplaza la verificacion del experto: la regla usa el mismo NDVI que el modelo y no
distingue palma de pasto, arvenses o bosque, que tienen NDVI parecido.

El Excel de salida incluye la zona del modelo: no se envia al experto antes de la sesion.

Uso:
    python experiments/contraste_escala_experto.py <ortomosaico.tif> <carpeta_resultados> [--poligonos poligonos_grandes.geojson]
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.windows import Window
from shapely.geometry import mapping, shape
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.models.crop_profiles import FUENTE_CODIRECTOR, NDVI_RANGES_EXPERTO, range_for  # noqa: E402
from app.models.raster_processor import RasterProcessor  # noqa: E402

RANGOS = [r["clave"] for r in NDVI_RANGES_EXPERTO]
VENTANA_PUNTO_M = 1.0

F = "Arial"
F_BASE, F_BOLD = Font(name=F, size=10), Font(name=F, size=10, bold=True)
F_HEAD = Font(name=F, size=10, bold=True, color="FFFFFF")
F_TITLE = Font(name=F, size=14, bold=True, color="1F3D2B")
F_NOTE = Font(name=F, size=9, italic=True, color="555555")
FILL_HEAD = PatternFill("solid", fgColor="2F5D3A")
FILL_SI, FILL_NO = PatternFill("solid", fgColor="D9EAD3"), PatternFill("solid", fgColor="F4CCCC")
BOX = Border(*(Side(style="thin", color="BFBFBF"),) * 4)
WRAP = Alignment(wrap_text=True, vertical="top")


def indices(stack, nodata):
    g, r, re, nir, alpha = (stack[i].astype("float64") for i in range(5))
    valid = (alpha > 0) & np.all(np.isfinite(stack[:4]), axis=0)
    if nodata is not None:
        valid &= np.all(stack[:4] != nodata, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {
            "ndvi": (nir - r) / (nir + r),
            "ndre": (nir - re) / (nir + re),
            "gndvi": (nir - g) / (nir + g),
        }
    valid &= np.isfinite(out["ndvi"]) & np.isfinite(out["ndre"]) & np.isfinite(out["gndvi"])
    return {k: v[valid] for k, v in out.items()}


def pct_por_rango(ndvi):
    # Mismos limites que range_for (min <= v < max), vectorizado para poligonos de millones de pixeles
    pos = np.searchsorted([r["min"] for r in NDVI_RANGES_EXPERTO[1:]], ndvi, side="right")
    n = np.bincount(pos, minlength=len(RANGOS))
    return {k: 100.0 * n[i] / max(ndvi.size, 1) for i, k in enumerate(RANGOS)}


def leer_bandas(ds):
    m = RasterProcessor._resolve_band_mapping(None, ds)
    return [m["green"], m["red"], m["rededge"], m["nir"], m.get("alpha", m["nir"])]


def contrastar_puntos(ds, bandas, puntos, clave, rango_zona):
    lado = int(round(VENTANA_PUNTO_M / abs(ds.res[0]))) | 1
    filas = []
    for f in sorted(puntos["features"], key=lambda f: f["properties"]["punto_id"]):
        pid = f["properties"]["punto_id"]
        x, y = f["geometry"]["coordinates"]
        row, col = ds.index(x, y)
        win = Window(col - lado // 2, row - lado // 2, lado, lado)
        stack = ds.read(bandas, window=win, boundless=True, fill_value=ds.nodata or 0)
        idx = indices(stack, ds.nodata)
        if idx["ndvi"].size == 0:
            continue
        ndvi = float(np.median(idx["ndvi"]))
        zona = clave[pid]
        regla = range_for(ndvi)["clave"]
        filas.append({
            "punto_id": pid, "zona": zona, "rango_modelo": rango_zona[zona],
            "ndvi": ndvi, "ndre": float(np.median(idx["ndre"])), "gndvi": float(np.median(idx["gndvi"])),
            "rango_regla": regla, "coincide": regla == rango_zona[zona],
        })
    return filas


def contrastar_poligonos(ds, bandas, poligonos, rango_zona):
    filas = []
    for f in sorted(poligonos["features"], key=lambda f: f["properties"]["poligono"]):
        pr = f["properties"]
        geom = shape(f["geometry"])
        arr, _ = rio_mask(ds, [mapping(geom)], crop=True, indexes=bandas, filled=True,
                          nodata=ds.nodata if ds.nodata is not None else 0)
        idx = indices(arr, ds.nodata)
        pct = pct_por_rango(idx["ndvi"])
        dominante = max(pct, key=pct.get)
        zona = pr["cluster_id"]
        filas.append({
            "poligono": pr["poligono"], "zona": zona, "rango_modelo": rango_zona[zona],
            "area_m2": pr["area_m2"], "ndvi": float(np.median(idx["ndvi"])),
            "ndre": float(np.median(idx["ndre"])), "gndvi": float(np.median(idx["gndvi"])),
            **{f"pct_{k}": pct[k] for k in RANGOS},
            "rango_regla": dominante, "pct_en_rango_modelo": pct[rango_zona[zona]],
            "coincide": dominante == rango_zona[zona],
        })
    return filas


def tabla(ws, r0, cabecera, filas, anchos=None):
    for j, h in enumerate(cabecera, start=1):
        c = ws.cell(row=r0, column=j, value=h)
        c.font, c.fill, c.border = F_HEAD, FILL_HEAD, BOX
        c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
    for i, fila in enumerate(filas, start=r0 + 1):
        for j, v in enumerate(fila, start=1):
            c = ws.cell(row=i, column=j, value=v)
            c.font, c.border = F_BASE, BOX
            if isinstance(v, float):
                c.number_format = "0.000" if abs(v) <= 1 else "0.0"
            if v in ("Sí", "No"):
                c.fill = FILL_SI if v == "Sí" else FILL_NO
    for j, w in enumerate(anchos or [], start=1):
        ws.column_dimensions[ws.cell(row=r0, column=j).column_letter].width = w
    ws.freeze_panes = ws.cell(row=r0 + 1, column=1)
    return r0 + len(filas) + 2


def excel(path, zonas, puntos, poligonos):
    wb = Workbook()
    ws = wb.active
    ws.title = "Leeme"
    ws["A1"] = "Contraste de la reflectancia con la escala del codirector"
    ws["A1"].font = F_TITLE
    ws["A2"] = "Motor GeoAI Analítico M2M · experiments/contraste_escala_experto.py"
    ws["A2"].font = F_NOTE
    textos = [
        ("Qué es", "La escala de NDVI del Acta N.° 01 aplicada automáticamente a la imagen: en cada punto de "
                   "verificación (mediana en una ventana de 1 × 1 m) y en cada polígono grande (porcentaje de "
                   "píxeles en cada rango). Se compara con el rango que el modelo le asignó a la zona."),
        ("Qué mide", "Consistencia interna: si el mapa de zonas respeta la regla de umbrales del experto."),
        ("Qué NO es", "No es la verificación del experto ni la exactitud temática. No llena las celdas amarillas "
                      "de los formatos de validación, que siguen en blanco para Cristian."),
        ("Por qué no la reemplaza", "1) La regla usa el mismo NDVI que el modelo: compararlos es medir el "
                                    "modelo contra sí mismo (circularidad). 2) Palma, pasto, arvenses y bosque "
                                    "tienen NDVI mayor a 0,6: la regla no sabe qué cobertura hay. Eso solo lo "
                                    "dice la interpretación visual del experto o la visita de campo."),
        ("Fuente de la escala", FUENTE_CODIRECTOR),
        ("Confidencial", "Contiene la zona del modelo en cada punto: NO enviar al experto antes de la sesión."),
    ]
    for i, (a, b) in enumerate(textos, start=4):
        ca, cb = ws.cell(row=i, column=1, value=a), ws.cell(row=i, column=2, value=b)
        ca.font, cb.font = F_BOLD, F_BASE
        ca.alignment = cb.alignment = WRAP
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 100
    ws.sheet_view.showGridLines = False

    ws = wb.create_sheet("Resumen")
    r = tabla(ws, 1, ["Zona del modelo", "Rango del modelo", "Área (ha)", "NDVI medio", "NDRE medio",
                      "Puntos", "% puntos que la regla pone en el mismo rango"],
              [[z["cluster_id"], z["rango"], z["area_ha"], z["ndvi_mean"], z["ndre_mean"],
                sum(p["zona"] == z["cluster_id"] for p in puntos),
                100.0 * np.mean([p["coincide"] for p in puntos if p["zona"] == z["cluster_id"]] or [0])]
               for z in zonas], [16, 20, 11, 12, 12, 10, 26])
    ws.cell(row=r, column=1, value="Puntos: zona del modelo (filas) frente al rango de la regla (columnas)").font = F_BOLD
    cruce = Counter((p["zona"], p["rango_regla"]) for p in puntos)
    r = tabla(ws, r + 1, ["Zona"] + RANGOS + ["Total"],
              [[z["cluster_id"]] + [cruce.get((z["cluster_id"], k), 0) for k in RANGOS]
               + [sum(cruce.get((z["cluster_id"], k), 0) for k in RANGOS)] for z in zonas])
    acuerdo_pt = 100.0 * np.mean([p["coincide"] for p in puntos])
    lineas = [f"Acuerdo en puntos: {acuerdo_pt:.1f} % de {len(puntos)}."]
    if poligonos:
        lineas.append(f"Acuerdo en polígonos (rango dominante): {100.0 * np.mean([p['coincide'] for p in poligonos]):.1f} % "
                      f"de {len(poligonos)}.")
    mismo = [z["cluster_id"] for z in zonas if z["rango"] == "buena_condicion"]
    if len(mismo) > 1:
        lineas.append(f"Las zonas {', '.join(map(str, mismo))} caen en el mismo rango de la escala (buena condición): "
                      "la regla no las separa; su diferencia está en NDRE. Qué cobertura hay en cada una lo "
                      "define el experto.")
    for i, t in enumerate(lineas, start=r):
        c = ws.cell(row=i, column=1, value=t)
        c.font = F_BOLD
    ws.freeze_panes = None

    ws = wb.create_sheet("Puntos")
    tabla(ws, 1, ["Punto", "Zona del modelo", "Rango del modelo", "NDVI (mediana 1 m)", "NDRE", "GNDVI",
                  "Rango según la escala", "¿Coincide?"],
          [[p["punto_id"], p["zona"], p["rango_modelo"], p["ndvi"], p["ndre"], p["gndvi"], p["rango_regla"],
            "Sí" if p["coincide"] else "No"] for p in puntos], [8, 14, 18, 16, 10, 10, 22, 11])
    if poligonos:
        ws = wb.create_sheet("Poligonos")
        tabla(ws, 1, ["Polígono", "Zona del modelo", "Rango del modelo", "Área (m²)", "NDVI (mediana)", "NDRE",
                      "GNDVI"] + [f"% {k}" for k in RANGOS] + ["Rango dominante", "% en el rango del modelo",
                                                              "¿Coincide?"],
              [[p["poligono"], p["zona"], p["rango_modelo"], p["area_m2"], p["ndvi"], p["ndre"], p["gndvi"]]
               + [p[f"pct_{k}"] for k in RANGOS] + [p["rango_regla"], p["pct_en_rango_modelo"],
                                                    "Sí" if p["coincide"] else "No"] for p in poligonos],
              [10, 14, 18, 11, 14, 9, 9, 12, 10, 11, 12, 18, 14, 11])
    wb.save(path)
    return lineas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tif")
    ap.add_argument("carpeta")
    ap.add_argument("--poligonos")
    ap.add_argument("--salida", default="contraste_escala_experto.xlsx")
    a = ap.parse_args()

    carpeta = Path(a.carpeta)
    res = json.loads((carpeta / "resultado.json").read_text(encoding="utf-8"))
    zonas = sorted((z for z in res["kmeans_zones"] if not z.get("is_noise")), key=lambda z: -z["ndvi_mean"])
    for z in zonas:
        z["rango"] = range_for(z["ndvi_mean"])["clave"]
    rango_zona = {z["cluster_id"]: z["rango"] for z in zonas}
    with open(carpeta / "clave_puntos_zona.csv", encoding="utf-8") as fh:
        clave = {int(r["punto_id"]): int(r["zona"]) for r in csv.DictReader(fh)}
    pts_gj = json.loads((carpeta / "puntos_verificacion.geojson").read_text(encoding="utf-8"))
    pol_gj = json.loads(Path(a.poligonos).read_text(encoding="utf-8")) if a.poligonos else None

    with rasterio.open(a.tif) as ds:
        bandas = leer_bandas(ds)
        puntos = contrastar_puntos(ds, bandas, pts_gj, clave, rango_zona)
        poligonos = contrastar_poligonos(ds, bandas, pol_gj, rango_zona) if pol_gj else []

    print(f"Escala: {FUENTE_CODIRECTOR}")
    print(f"{'zona':>4} {'rango del modelo':<20} {'NDVI':>6} {'NDRE':>6} {'puntos':>6} {'coinciden':>9}")
    for z in zonas:
        pz = [p for p in puntos if p["zona"] == z["cluster_id"]]
        print(f"{z['cluster_id']:>4} {z['rango']:<20} {z['ndvi_mean']:>6.3f} {z['ndre_mean']:>6.3f} "
              f"{len(pz):>6} {sum(p['coincide'] for p in pz):>9}")
    if poligonos:
        print(f"{'pol':>4} {'zona':>4} {'rango del modelo':<20} {'NDVI':>6} {'% en rango modelo':>18} {'dominante':<20}")
        for p in poligonos:
            print(f"{p['poligono']:>4} {p['zona']:>4} {p['rango_modelo']:<20} {p['ndvi']:>6.3f} "
                  f"{p['pct_en_rango_modelo']:>18.1f} {p['rango_regla']:<20}")
    for t in excel(carpeta / a.salida, zonas, puntos, poligonos):
        print(t)
    print(f"Excel: {carpeta / a.salida}")


if __name__ == "__main__":
    main()
