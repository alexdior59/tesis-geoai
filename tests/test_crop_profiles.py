import pytest

from app.models.crop_profiles import CROP_PROFILES, NDVI_RANGES_EXPERTO, get_profile, range_for


def test_rangos_contiguos_sin_huecos():
    for a, b in zip(NDVI_RANGES_EXPERTO, NDVI_RANGES_EXPERTO[1:]):
        assert a["max"] == b["min"]
    assert NDVI_RANGES_EXPERTO[0]["min"] == -1.0
    assert NDVI_RANGES_EXPERTO[-1]["max"] > 1.0


@pytest.mark.parametrize("ndvi, clave", [
    (0.05, "suelo_o_senescencia"),
    (0.2, "estres"),
    (0.45, "estres"),
    (0.55, "transicion"),
    (0.6, "buena_condicion"),
    (1.0, "buena_condicion"),
])
def test_escala_del_codirector(ndvi, clave):
    assert range_for(ndvi)["clave"] == clave


def test_perfil_palma_trae_parametros_de_siembra():
    p = get_profile("palma")
    assert p["palm_analysis"]["spacing_m"] == 9.0
    assert p["palm_analysis"]["ndvi_alert"] == 0.5
    assert p["inspection_range"] == (0.2, 0.5)
    assert "Acta" in p["fuente"]


def test_cultivo_no_soportado():
    with pytest.raises(ValueError):
        get_profile("banano")
    assert set(CROP_PROFILES) == {"generico", "palma"}
