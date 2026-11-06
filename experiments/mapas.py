"""
Mapas en PNG a partir de la carpeta que genera analisis_completo.py: zonas FCM, transicion entre
zonas, incertidumbre, zonas de inspeccion con los patrones DBSCAN y, en palma, el estado de cada
palma. Si se pasa el ortomosaico, se usa de fondo en falso color (NIR, rojo, verde). Si el analisis
se hizo con el poligono del lote, se dibuja su limite y los mapas se encuadran en el.

Uso:
    python experiments/mapas.py <carpeta_resultados> [--tif ortomosaico.tif] [--dpi 200]
"""
import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.models.crop_profiles import range_for  # noqa: E402
from app.views.response_serializer import NODATA_CODE, TRANSITION_CODE  # noqa: E402

MAX_LADO_PX = 2500
COLOR_TRANSICION = "#bdbdbd"
NOMBRE_RANGO = {
    "buena_condicion": "buena condición",
    "transicion": "transición (NDVI)",
    "estres": "estrés",
    "suelo_o_senescencia": "suelo o senescencia",
}
COLOR_PALMA = {
    "normal": "#ffe14d",
    "verificar_copa": "#ff9d1a",
    "inspeccionar": "#e8231e",
    "posible_faltante": "#00e5ff",
}
NOMBRE_PALMA = {
    "normal": "Normal",
    "verificar_copa": "Copa débil (verificar)",
    "inspeccionar": "Inspeccionar (NDVI)",
    "posible_faltante": "Posible faltante",
}


def fmt(valor, decimales=2):
    """Numero con coma decimal, como se escribe en espanol."""
    return f"{valor:.{decimales}f}".replace(".", ",")


def leer_raster(path, categorico):
    """Lee la primera banda reducida para dibujar; devuelve el arreglo y su extensión en metros."""
    with rasterio.open(path) as ds:
        f = max(1, int(np.ceil(max(ds.width, ds.height) / MAX_LADO_PX)))
        forma = (ds.height // f, ds.width // f)
        metodo = Resampling.mode if categorico else Resampling.average
        datos = ds.read(1, out_shape=forma, resampling=metodo)
        b = ds.bounds
        return datos, (b.left, b.right, b.bottom, b.top)


def fondo_falso_color(tif):
    """Composición NIR-rojo-verde estirada entre los percentiles 2 y 98, reducida para dibujar."""
    with rasterio.open(tif) as ds:
        f = max(1, int(np.ceil(max(ds.width, ds.height) / MAX_LADO_PX)))
        forma = (ds.height // f, ds.width // f)
        nombres = [(d or "").lower().replace(" ", "") for d in ds.descriptions]
        idx = {n: i + 1 for i, n in enumerate(nombres)}
        bandas = [idx.get("nir", 4), idx.get("red", 2), idx.get("green", 1)]
        rgb = ds.read(bandas, out_shape=(3, *forma), resampling=Resampling.average).astype(np.float32)
        if ds.nodata is not None:
            rgb[rgb == ds.nodata] = np.nan
        b = ds.bounds
    rgb[rgb <= 0] = np.nan
    for i in range(3):
        lo, hi = np.nanpercentile(rgb[i], (2, 98))
        rgb[i] = np.clip((rgb[i] - lo) / max(hi - lo, 1e-6), 0, 1)
    alfa = np.isfinite(rgb).all(axis=0).astype(np.float32)
    imagen = np.dstack([np.nan_to_num(rgb[i]) for i in range(3)] + [alfa])
    return imagen, (b.left, b.right, b.bottom, b.top)


def colores_por_vigor(zonas):
    """Verde oscuro para la zona de mayor NDVI y rojo para la de menor."""
    orden = sorted(zonas, key=lambda z: -z["ndvi_mean"])
    paleta = plt.get_cmap("RdYlGn")(np.linspace(0.92, 0.08, len(orden)))
    return {z["cluster_id"]: paleta[i] for i, z in enumerate(orden)}


def ndvi_de_zona(res, cluster_id):
    """NDVI medio de la zona, para ordenar las leyendas igual que los colores."""
    return next(z["ndvi_mean"] for z in res["kmeans_zones"] if z["cluster_id"] == cluster_id)


def etiqueta_zona(z, total_ha):
    nombre = z.get("expert_label") or NOMBRE_RANGO.get(range_for(z["ndvi_mean"])["clave"], "")
    pct = z["area_ha"] * 100 / total_ha if total_ha else 0
    return f"Zona {z['cluster_id']}: {nombre} · NDVI {fmt(z['ndvi_mean'])} · {pct:.0f} %"


def preparar_ejes(ax, extension, titulo, lote=None):
    if lote is not None:
        lote.boundary.plot(ax=ax, color="black", linewidth=1.2, linestyle="--")
        x0, y0, x1, y1 = lote.total_bounds
        margen = 0.03 * max(x1 - x0, y1 - y0)
        extension = (x0 - margen, x1 + margen, y0 - margen, y1 + margen)
    ax.set_xlim(extension[0], extension[1])
    ax.set_ylim(extension[2], extension[3])
    ax.set_aspect("equal")
    ax.set_title(titulo, fontsize=12, loc="left")
    ax.set_xlabel("Este (m, UTM 18N)", fontsize=8)
    ax.set_ylabel("Norte (m, UTM 18N)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.ticklabel_format(useOffset=False, style="plain")
    barra_de_escala(ax, extension)
    flecha_norte(ax)


def barra_de_escala(ax, extension):
    ancho = extension[1] - extension[0]
    largo = next(v for v in (500, 200, 100, 50, 20, 10, 5) if v <= ancho / 4)
    x0 = extension[0] + ancho * 0.04
    y0 = extension[2] + (extension[3] - extension[2]) * 0.04
    ax.plot([x0, x0 + largo], [y0, y0], color="black", lw=4, solid_capstyle="butt")
    ax.plot([x0, x0 + largo], [y0, y0], color="white", lw=2, solid_capstyle="butt")
    ax.text(x0 + largo / 2, y0, f"{largo} m", ha="center", va="bottom", fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1})


def flecha_norte(ax):
    ax.annotate("N", xy=(0.96, 0.95), xytext=(0.96, 0.84), xycoords="axes fraction", ha="center",
                fontsize=10, fontweight="bold",
                arrowprops={"facecolor": "black", "width": 3, "headwidth": 9})


def guardar(fig, path, dpi):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def mapa_zonas(carpeta, res, fondo, dpi, salida, lote):
    zonas = gpd.read_file(carpeta / "zonas_fcm.geojson")
    colores = colores_por_vigor(res["kmeans_zones"])
    total = sum(z["area_ha"] for z in res["kmeans_zones"])
    fig, ax = plt.subplots(figsize=(9, 8))
    extension = tuple(zonas.total_bounds[[0, 2, 1, 3]])
    zonas.plot(ax=ax, color=[colores[c] for c in zonas["cluster_id"]], linewidth=0)
    m = res["quality_metrics"]["kmeans"]
    preparar_ejes(ax, extension, f"Zonas de manejo (FCM, k = {m['n_clusters']})", lote)
    leyenda = [Patch(color=colores[z["cluster_id"]], label=etiqueta_zona(z, total))
               for z in sorted(res["kmeans_zones"], key=lambda z: -z["ndvi_mean"])]
    if lote is not None:
        area = res["data_provenance"].get("lote_area_ha")
        texto = f"Límite del lote ({fmt(area, 2)} ha)" if area else "Límite del lote"
        leyenda.append(plt.Line2D([], [], color="black", linestyle="--", label=texto))
    ax.legend(handles=leyenda, loc="upper left", fontsize=8, framealpha=0.9)
    ax.text(0.99, 0.01, f"FPC {fmt(m['fcm_partition_coefficient'])} · Silhouette {fmt(m['silhouette_score'])}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    guardar(fig, salida / "01_zonas_fcm.png", dpi)


def mapa_transicion(carpeta, res, dpi, salida, lote):
    path = carpeta / "zonas_transicion.tif"
    if not path.exists():
        return
    datos, extension = leer_raster(path, categorico=True)
    colores = colores_por_vigor(res["kmeans_zones"])
    ids = sorted(colores)
    tabla = np.full(256, len(ids) + 1, dtype=np.int16)
    for i, cid in enumerate(ids):
        tabla[cid] = i
    tabla[TRANSITION_CODE] = len(ids)
    indices = np.ma.masked_where(datos == NODATA_CODE, tabla[datos])
    cmap = ListedColormap([colores[c] for c in ids] + [COLOR_TRANSICION, "white"])

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(indices, extent=extension, cmap=cmap, vmin=0, vmax=len(ids) + 1, interpolation="nearest")
    t = res.get("transicion") or {}
    preparar_ejes(ax, extension, f"Transición entre zonas: {fmt(t.get('pct_pixeles_transicion', 0), 1)} % de los píxeles", lote)
    leyenda = [
        Patch(color=colores[z["cluster_id"]], label=f"Zona {z['cluster_id']} · {fmt(z['pct_transicion'], 1)} % en transición")
        for z in sorted(t.get("por_zona", []), key=lambda z: -ndvi_de_zona(res, z["cluster_id"]))
    ]
    umbral = fmt(t.get("umbral_pertenencia", 0.6), 1)
    leyenda.append(Patch(color=COLOR_TRANSICION, label=f"Transición (pertenencia < {umbral})"))
    ax.legend(handles=leyenda, loc="upper left", fontsize=8, framealpha=0.9)
    guardar(fig, salida / "02_transicion.png", dpi)


def mapa_incertidumbre(carpeta, dpi, salida, lote):
    path = carpeta / "incertidumbre.tif"
    if not path.exists():
        return
    with rasterio.open(path) as ds:
        f = max(1, int(np.ceil(max(ds.width, ds.height) / MAX_LADO_PX)))
        # se promedia ignorando los pixeles sin dato para no oscurecer los bordes
        datos = ds.read(1, out_shape=(ds.height // f, ds.width // f), resampling=Resampling.average,
                        masked=True)
        b = ds.bounds
    extension = (b.left, b.right, b.bottom, b.top)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(datos, extent=extension, cmap="RdYlGn", vmin=25, vmax=100, interpolation="nearest")
    preparar_ejes(ax, extension, "Pertenencia de cada píxel a su zona (%)", lote)
    barra = fig.colorbar(im, ax=ax, shrink=0.7)
    barra.set_label("Pertenencia máxima FCM (%)", fontsize=8)
    guardar(fig, salida / "03_incertidumbre.png", dpi)


def mapa_inspeccion(carpeta, res, fondo, dpi, salida, lote):
    insp = gpd.read_file(carpeta / "zonas_inspeccion.geojson")
    patrones_path = carpeta / "patrones_dbscan.geojson"
    patrones = gpd.read_file(patrones_path) if patrones_path.exists() else None
    zonas = gpd.read_file(carpeta / "zonas_fcm.geojson")
    extension = tuple(zonas.total_bounds[[0, 2, 1, 3]])

    fig, ax = plt.subplots(figsize=(9, 8))
    if fondo is not None:
        ax.imshow(fondo[0], extent=fondo[1])
    leyenda = []
    if len(insp):
        insp.plot(ax=ax, facecolor=(1, 0, 0, 0.35), edgecolor="red", linewidth=1)
        leyenda.append(Patch(facecolor=(1, 0, 0, 0.35), edgecolor="red",
                             label=f"Zona de inspección (NDVI 0,2–0,5): {len(insp)}"))
    if patrones is not None and len(patrones):
        patrones.plot(ax=ax, facecolor="none", edgecolor="#00e5ff", linewidth=0.8)
        leyenda.append(Patch(facecolor="none", edgecolor="#00e5ff", label=f"Patrón DBSCAN: {len(patrones)}"))
    c = res.get("condicion_por_rangos") or {}
    titulo = "Zonas de inspección y patrones espaciales"
    if c:
        titulo += f" ({fmt(c['inspeccion']['pct_area_en_rango'], 1)} % del área en NDVI 0,2–0,5)"
    preparar_ejes(ax, extension, titulo, lote)
    if leyenda:
        ax.legend(handles=leyenda, loc="upper left", fontsize=8, framealpha=0.9)
    guardar(fig, salida / "04_inspeccion_y_patrones.png", dpi)


def mapa_palmas(carpeta, res, fondo, dpi, salida, lote):
    path = carpeta / "palmas.geojson"
    if not path.exists():
        return
    palmas = gpd.read_file(path)
    zonas = gpd.read_file(carpeta / "zonas_fcm.geojson")
    extension = tuple(zonas.total_bounds[[0, 2, 1, 3]])
    fig, ax = plt.subplots(figsize=(9, 8))
    if fondo is not None:
        ax.imshow(fondo[0], extent=fondo[1])
    leyenda = []
    for estado in ("normal", "verificar_copa", "inspeccionar", "posible_faltante"):
        grupo = palmas[palmas["estado"] == estado]
        if not len(grupo):
            continue
        marcador = "x" if estado == "posible_faltante" else "o"
        tam = 14 if estado == "posible_faltante" else (5 if estado == "normal" else 12)
        grupo.plot(ax=ax, color=COLOR_PALMA[estado], markersize=tam, marker=marcador, linewidth=1)
        leyenda.append(plt.Line2D([], [], color=COLOR_PALMA[estado], marker=marcador, linestyle="",
                                  label=f"{NOMBRE_PALMA[estado]}: {len(grupo):,}".replace(",", ".")))
    p = res.get("analisis_palmas") or {}
    titulo = "Estado de cada palma"
    if p:
        conteo = f"{p['conteo_palmas']:,}".replace(",", ".")
        titulo += f" ({conteo} palmas, {fmt(p['densidad_palmas_ha'], 1)} por ha)"
    preparar_ejes(ax, extension, titulo, lote)
    ax.legend(handles=leyenda, loc="upper left", fontsize=8, framealpha=0.9)
    guardar(fig, salida / "05_palmas.png", dpi)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("carpeta", help="carpeta generada por analisis_completo.py")
    ap.add_argument("--tif", help="ortomosaico para el fondo en falso color")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()

    carpeta = Path(a.carpeta)
    res = json.loads((carpeta / "resultado.json").read_text(encoding="utf-8"))
    salida = carpeta / "mapas"
    salida.mkdir(exist_ok=True)
    fondo = fondo_falso_color(a.tif) if a.tif else None
    lote_path = carpeta / "lote.geojson"
    lote = gpd.read_file(lote_path) if lote_path.exists() else None

    print(f"Mapas en {salida}:")
    mapa_zonas(carpeta, res, fondo, a.dpi, salida, lote)
    mapa_transicion(carpeta, res, a.dpi, salida, lote)
    mapa_incertidumbre(carpeta, a.dpi, salida, lote)
    mapa_inspeccion(carpeta, res, fondo, a.dpi, salida, lote)
    mapa_palmas(carpeta, res, fondo, a.dpi, salida, lote)


if __name__ == "__main__":
    main()
