FUENTE_CODIRECTOR = "Concepto del codirector (area agronomica), Acta de reunion N. 01 del 17/10/2026"

NDVI_RANGES_EXPERTO = [
    {"clave": "suelo_o_senescencia", "min": -1.0, "max": 0.2,
     "descripcion": "Suelo desnudo o cultivo en senescencia total"},
    {"clave": "estres", "min": 0.2, "max": 0.5,
     "descripcion": "Cultivo con estres: zona de inspeccion"},
    {"clave": "transicion", "min": 0.5, "max": 0.6,
     "descripcion": "Transicion entre estres y buena condicion"},
    {"clave": "buena_condicion", "min": 0.6, "max": 1.0001,
     "descripcion": "Vegetacion en buena condicion (subrangos requieren calibracion por cultivo)"},
]

CROP_PROFILES = {
    "generico": {
        "nombre": "Cultivo generico",
        "ndvi_ranges": NDVI_RANGES_EXPERTO,
        "inspection_range": (0.2, 0.5),
        "palm_analysis": None,
        "fuente": FUENTE_CODIRECTOR,
    },
    "palma": {
        "nombre": "Palma de aceite",
        "ndvi_ranges": NDVI_RANGES_EXPERTO,
        "inspection_range": (0.2, 0.5),
        "palm_analysis": {"spacing_m": 9.0, "ndvi_alert": 0.5},
        "fuente": FUENTE_CODIRECTOR,
    },
}


def get_profile(crop_type: str) -> dict:
    if crop_type not in CROP_PROFILES:
        raise ValueError(
            f"Cultivo no soportado: {crop_type}. Opciones: {', '.join(CROP_PROFILES)}"
        )
    return CROP_PROFILES[crop_type]


def range_for(ndvi_value: float, ranges=NDVI_RANGES_EXPERTO) -> dict:
    for r in ranges:
        if r["min"] <= ndvi_value < r["max"]:
            return r
    return ranges[0] if ndvi_value < ranges[0]["min"] else ranges[-1]
