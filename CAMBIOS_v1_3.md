# Cambios y pruebas de la versión 1.3 del Motor GeoAI Analítico M2M

## 1. Propósito y base de comparación

Este informe describe los cambios hechos sobre la v1.2 (ver `CAMBIOS_v1_2.md`), el fundamento de cada uno y las pruebas que los verifican. La v1.3 no cambia el método: con los mismos datos asigna cada píxel a la misma zona que la v1.2. Corrige problemas de trazabilidad y robustez, y agrega la categoría de transición entre zonas.

| Versión | Contenido |
|---|---|
| 1.2 | Línea base congelada y persistente; criterio del experto; análisis por palma; explicabilidad; incertidumbre; exactitud temática; 56 pruebas |
| 1.3 (este informe) | Trazabilidad del resultado; línea base con su umbral de NDVI; área por polígono; patrones DBSCAN en GeoJSON; carga del archivo por bloques; transición entre zonas; mapas en PNG; recorte con el polígono del lote; 79 pruebas |

## 2. Cambios en el código y su fundamento

### 2.1 Trazabilidad y línea base

| Cambio | Por qué |
|---|---|
| El `job_id` del JSON y del resumen es el mismo de la API | En la v1.2 el serializador generaba un identificador nuevo, distinto del que el usuario usa para consultar el resultado |
| `analysis_params` incluye el umbral de NDVI (`vegetation_threshold`) y el cultivo | Sin el umbral no se puede reproducir un análisis: el umbral decide qué píxeles son vegetación |
| La línea base guarda su umbral de NDVI y los vuelos que se comparan con ella lo usan, igual que su k | Con otro umbral cambian los píxeles analizados y el cambio por zona deja de ser comparable |
| `source_file` sin el prefijo interno del job | Que el resultado muestre el nombre real del archivo |

### 2.2 Salidas geográficas

| Cambio | Por qué |
|---|---|
| En el GeoJSON de zonas, `area_ha` es el área de cada polígono y `zone_area_ha` la de toda la zona | En la v1.2 cada polígono mostraba el área total de su zona, lo que confundía la tabla de atributos en QGIS |
| Nuevo endpoint `/result/{id}/patterns`: patrones DBSCAN como polígonos, uno por patrón | Los patrones se calculaban pero no tenían salida geográfica |
| Las áreas se calculan con la resolución sin redondear | La resolución se redondeaba a 4 decimales antes de calcular áreas |

### 2.3 Robustez de la API

| Cambio | Por qué |
|---|---|
| El archivo se escribe a disco por bloques de 16 MB | La v1.2 cargaba el ortomosaico completo (hasta 1,5 GB) en memoria antes de guardarlo |
| Un análisis a la vez por defecto (`GEOAI_MAX_CONCURRENT_JOBS`); los demás quedan en cola | Un ortomosaico completo ocupa varios GB de memoria; dos al tiempo pueden agotarla |
| Del nombre del archivo subido se toma solo el nombre, sin rutas | Un nombre con `../` podía escribir fuera de la carpeta de cargas |
| Mientras el análisis no termina, los endpoints de resultado responden 409 | La v1.2 respondía 202 como si fuera un error, lo que confunde a un cliente |
| La banda Alpha también se reconoce por su *color interpretation*, y "Infrared" ya no se confunde con rojo | Exportaciones que nombran las bandas de otra forma |

### 2.4 Transición entre zonas

| Cambio | Por qué |
|---|---|
| Cada píxel guarda su segunda zona más cercana | Saber con qué zona se mezcla un píxel ambiguo |
| Píxel de transición: pertenencia máxima menor a 0,6 (el mismo umbral de la explicabilidad) | Señalar dónde la asignación es poco segura sin cambiar el mapa de zonas |
| Sección `transicion` en el JSON y en el resumen: porcentaje y área de cada zona en transición, y con qué zonas se mezcla | Medir qué tan definida está cada zona |
| Nuevo endpoint `/result/{id}/zones-map`: GeoTIFF con el número de zona y 200 para la transición | Ver la transición en QGIS |
| La transición no se vectoriza | A 6 cm por píxel está casi toda en el borde de las copas; como polígonos serían cientos de fragmentos de menos de 1 m² |
| FCM ya no arranca con dos centroides iguales | Con muchos píxeles idénticos, el arranque al azar podía repetir un valor y FCM no separaba esas dos zonas. Encontrado por las pruebas nuevas; no afecta datos reales |

### 2.5 Mapas

| Cambio | Por qué |
|---|---|
| Nuevo `experiments/mapas.py`: zonas FCM, transición, incertidumbre, zonas de inspección con patrones DBSCAN y estado de cada palma, en PNG, con escala, norte y leyenda | Ver el resultado sin abrir QGIS y tener figuras para el documento de tesis |
| Las capas vectoriales se dibujan con geopandas y las ráster con rasterio, reducidas a unos 2 500 píxeles de lado | Que los mapas del lote completo se generen en segundos |
| Con `--tif`, el ortomosaico se usa de fondo en falso color (NIR, rojo, verde) | Ubicar las zonas de inspección y las palmas sobre la imagen |
| `matplotlib` en `requirements-dev.txt` | Lo usa el script de mapas; la API no lo necesita |
| El reporte de transición ya no lista mezclas que redondean a 0 % | Eran unos pocos píxeles y confundían la lectura |

### 2.6 Polígono del lote

| Cambio | Por qué |
|---|---|
| Nuevo `app/models/aoi.py`: lee el polígono (GeoJSON, shapefile o GeoPackage), valida que tenga polígonos y sistema de coordenadas, y lo reproyecta al del ortomosaico | El ortomosaico incluye áreas que no son el lote; en PALMA, la zona de estrés era casi toda una franja sin palma |
| `RasterProcessor` deja sin dato los píxeles fuera del polígono | Así todo lo demás (rangos, zonas, palmas, transición, puntos de verificación) queda dentro del lote sin cambiar cada módulo |
| Parámetro opcional `aoi` en `POST /analyze` (archivo GeoJSON); `lote_area_ha` en `data_provenance` y en el resumen | Enviar el lote desde la API |
| La línea base guarda el polígono y los vuelos que se comparan con ella lo usan; enviar otro da 422 | Comparar dos vuelos sobre la misma área |
| `--lote` en `analisis_completo.py` y `seleccion_k.py`; `mapas.py` dibuja el límite y encuadra los mapas en él | Usar el recorte en los experimentos |

### 2.7 Otros archivos

| Archivo | Cambio |
|---|---|
| `experiments/analisis_completo.py` | Usa la resolución sin redondear y guarda también `zonas_transicion.tif` y `patrones_dbscan.geojson` |
| `resultados_referencia/muestra/` y `muestra_consola.txt` | Regenerados con la v1.3. Los 120 puntos de verificación (`clave_puntos_zona.csv`) son idénticos a los de la v1.2 |
| `resultados_referencia/pruebas_pytest.txt` | Reporte de las 79 pruebas |
| `resultados_referencia/palma_resumen.txt` y `mapas_palma/` | Análisis del lote completo repetido con la v1.3, y sus mapas |
| `app/main.py` y `response_serializer.py` | Versión de la API 1.3.0 |
| `docker-compose.yml` | Variable `GEOAI_MAX_CONCURRENT_JOBS` y reinicio automático del contenedor |
| `accuracy.py`, `condition_report.py`, `palm_analyzer.py` | Las coordenadas se calculan con `transform @ (...)` en lugar de `transform * (...)`, que la librería affine 3 marca como obsoleto |
| `requirements.txt` y `pytest.ini` | Versión fija de affine (3.0.1) y filtros para los avisos que vienen de librerías externas (rasterio y el cliente de pruebas de FastAPI) |
| `.gitignore` | También se versionan los GeoTIFF de `resultados_referencia/` (mapa de incertidumbre y de zonas de la muestra) |
| `README_PROFESOR.md` y `METODOLOGIA.md` | Endpoints nuevos, sección de transición y resultados en la muestra |

## 3. Pruebas nuevas

Se suman 23 pruebas a las 56 de la v1.2, para un total de 79.

### 3.1 tests/test_robustez.py (9)

| Prueba | Qué verifica |
|---|---|
| test_alpha_detectada_por_color_interpretation | La banda Alpha se reconoce aunque su nombre no lo diga |
| test_infrared_no_se_confunde_con_red | "Near infrared" se asigna al NIR y no al rojo |
| test_umbral_por_constructor | El umbral de NDVI se pasa al procesador y falla con un mensaje claro si ningún píxel lo supera |
| test_resultado_trazable | El `job_id`, el umbral y el nombre del archivo del resultado coinciden con los de la solicitud |
| test_area_de_cada_poligono_y_de_la_zona | El área de cada polígono es la de su geometría y la suma cuadra con el área de las zonas |
| test_patrones_dbscan_en_geojson | El GeoJSON de patrones tiene un polígono por patrón detectado |
| test_linea_base_impone_su_umbral | La línea base guarda su umbral y el vuelo comparado lo usa aunque se envíe otro |
| test_resultado_pendiente_responde_409 | Un resultado que aún no está listo responde 409 |
| test_nombre_de_archivo_sin_rutas | Un nombre de archivo con rutas se reduce al nombre |

### 3.2 tests/test_transicion.py (6)

Usa un lote sintético que pasa poco a poco de palma vigorosa a estrés, de modo que se sabe dónde debe estar la transición.

| Prueba | Qué verifica |
|---|---|
| test_la_transicion_queda_dentro_del_degradado | Todos los píxeles de transición están en la franja del degradado y en las zonas puras la asignación es segura |
| test_segunda_zona_es_la_otra_condicion | La segunda zona de cada píxel es distinta de la asignada |
| test_reporte_de_transicion | Porcentaje y área de transición cuadran con los píxeles; cada zona se mezcla solo con la otra |
| test_pixeles_repetidos_no_bloquean_fcm | Sin ruido (miles de píxeles idénticos) FCM separa igual las dos condiciones |
| test_sin_degradado_casi_no_hay_transicion | En franjas bien separadas la transición es menor al 2 % |
| test_mapa_de_zonas_con_transicion_por_api | El GeoTIFF de `zones-map` tiene CRS, nodata y la clase 200 en la misma proporción que el reporte |

### 3.3 tests/test_mapas.py (2)

| Prueba | Qué verifica |
|---|---|
| test_mapas_de_un_analisis_de_palma | `analisis_completo.py` y `mapas.py` corren sobre la plantación sintética y generan los cinco PNG |
| test_mapas_con_el_poligono_del_lote | Con `--lote` se guarda el límite y los mapas se generan; sin `--lote` no queda el límite de una corrida anterior |

### 3.4 tests/test_lote.py (6)

| Prueba | Qué verifica |
|---|---|
| test_fuera_del_lote_queda_sin_dato | Fuera del polígono no hay dato y adentro sí; el área del lote es la del polígono |
| test_lote_en_otro_crs_se_reproyecta | El mismo polígono en WGS84 recorta lo mismo que en UTM |
| test_lote_que_no_se_cruza_con_la_imagen | Un polígono lejos de la imagen da un error claro |
| test_lote_sin_poligonos | Un archivo con puntos o que no es GeoJSON se rechaza |
| test_analisis_y_linea_base_con_el_lote | Por la API, con la mitad del lote los rangos pasan a 50 % y 50 %; la línea base guarda el lote, el vuelo siguiente lo usa y enviar otro da 422 |
| test_lote_invalido_por_api | Un GeoJSON sin geometrías da 422 |

## 4. Compatibilidad con la v1.2

- Con los mismos datos, cada píxel queda en la misma zona y los puntos de verificación son los mismos.
- Las áreas cambian desde el cuarto decimal por el cálculo con la resolución exacta (en la muestra, de 0,1695 a 0,1694 ha).
- Cambian para un cliente de la API: el significado de `area_ha` en el GeoJSON de zonas, la respuesta 409 en lugar de 202 mientras el análisis no termina, y el k y el umbral, que se toman de la línea base cuando se envía `reference_model_id`.
- Las líneas base guardadas con la v1.2 siguen funcionando; si no traen umbral, se usa el que se envíe.

## 5. Resultado en la muestra (90 × 90 m, k = 4)

| Zona | NDVI | % en transición | Se mezcla con |
|---|---|---|---|
| 3 | 0.881 | 13.2 % | zona 1 |
| 1 | 0.843 | 23.0 % | zona 0 (52 %) y zona 3 (48 %) |
| 0 | 0.743 | 28.9 % | zona 1 (90 %) y zona 2 (10 %) |
| 2 (estrés) | 0.254 | 1.7 % | zona 0 |

En total, el 15,1 % de los píxeles está en transición. Con k = 3 baja a 7,1 %. La zona de estrés es la mejor definida y las subzonas de buena condición forman un gradiente.

## 6. Lote completo con la v1.3

El análisis completo del lote PALMA (12 044 × 7 539 píxeles de 6 cm, 65.5 millones de píxeles de vegetación) se repitió con la v1.3: tardó 176 s con `analisis_completo.py` y 184 s por la API con un servidor uvicorn. La subida del archivo de 1,5 GB tomó 5 s sin pasar de 0,2 GB de memoria, y el análisis llegó a un pico de 5,4 GB. Las zonas, los rangos y el análisis por palma coinciden con la v1.2. Los mapas muestran que la zona FCM de estrés no es palma estresada: el 89 % de sus píxeles está en una franja sin palma del borde este del ortomosaico y el resto en las vías internas. Esa franja es también el 90 % del área de las zonas de inspección. Por eso se agregó el recorte con el polígono del lote (sección 2.6).

Para ver el efecto del recorte, mientras llega el polígono oficial, se armó un polígono aproximado con la envolvente de las palmas detectadas más 4,5 m (22,7 ha; `resultados_referencia/lote_aproximado/`). Con él:

| Medida | Sin polígono | Con el polígono aproximado |
|---|---|---|
| Área analizada | 24,1 ha con datos | 22,4 ha |
| NDVI 0,2–0,5 (zonas de inspección) | 2,24 %, 0,54 ha en 14 zonas | 0,93 %, 0,21 ha en 13 zonas |
| Zona FCM de menor NDVI | "estrés", NDVI 0,30, 0,58 ha | "transición", NDVI 0,57, 0,88 ha (vías y bordes de copa) |
| Palmas | 3.067 (23 a inspeccionar) | 3.053 (22 a inspeccionar) |
| Patrones DBSCAN | 1.075 (ruido 29 %) | 91 (ruido 2 %) |
| Píxeles de transición | 23,8 % | 27,3 % |

Dentro de la plantación no queda una zona de estrés: la zona de menor NDVI está en el rango de transición del codirector y corresponde sobre todo a vías y bordes de copa. Las palmas que necesitan revisión las identifica mejor el análisis por palma.

La selección de k también cambia (`lote_aproximado/seleccion_k_palma_lote_aproximado.txt`): sobre toda la imagen FPI y entropía tenían su mínimo en k = 3, pero dentro del lote el FPI lo tiene en k = 2 y la entropía en k = 4. La tercera zona de antes era la franja sin palma.

## 7. Cómo verificar este informe

- Pruebas: `python -m pytest -v` debe dar 79 aprobadas (reporte en `resultados_referencia/pruebas_pytest.txt`).
- Muestra: `python experiments/analisis_completo.py sample_data/palma_muestra.tif salida --cultivo palma` reproduce `resultados_referencia/muestra/`.

## 8. Pendientes

- Repetir con la v1.3 los demás experimentos del lote completo (selección de k, línea base, explicabilidad y escala).
- Conseguir el polígono oficial del lote PALMA (de la finca o dibujado en QGIS) y repetir el análisis con él.
- Decidir el número de zonas con el polígono oficial y con el experto: dentro del lote los índices no señalan un k claro. Hay que decidirlo antes de generar los formatos de validación, porque los puntos se sortean por zona.
- Al evaluar la exactitud temática, reportarla por separado para puntos seguros y de transición.
- Probar la imagen Docker.
