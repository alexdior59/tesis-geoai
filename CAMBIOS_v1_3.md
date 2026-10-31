# Cambios y pruebas de la versión 1.3 del Motor GeoAI Analítico M2M

## 1. Propósito y base de comparación

Este informe describe los cambios hechos sobre la v1.2 (ver `CAMBIOS_v1_2.md`), el fundamento de cada uno y las pruebas que los verifican. La v1.3 no cambia el método: con los mismos datos asigna cada píxel a la misma zona que la v1.2. Corrige problemas de trazabilidad y robustez, y agrega la categoría de transición entre zonas.

| Versión | Contenido |
|---|---|
| 1.2 | Línea base congelada y persistente; criterio del experto; análisis por palma; explicabilidad; incertidumbre; exactitud temática; 56 pruebas |
| 1.3 (este informe) | Trazabilidad del resultado; línea base con su umbral de NDVI; área por polígono; patrones DBSCAN en GeoJSON; carga del archivo por bloques; transición entre zonas; 71 pruebas |

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

### 2.5 Otros archivos

| Archivo | Cambio |
|---|---|
| `experiments/analisis_completo.py` | Usa la resolución sin redondear y guarda también `zonas_transicion.tif` y `patrones_dbscan.geojson` |
| `resultados_referencia/muestra/` y `muestra_consola.txt` | Regenerados con la v1.3. Los 120 puntos de verificación (`clave_puntos_zona.csv`) son idénticos a los de la v1.2 |
| `resultados_referencia/pruebas_pytest.txt` | Reporte de las 71 pruebas |
| `app/main.py` y `response_serializer.py` | Versión de la API 1.3.0 |
| `docker-compose.yml` | Variable `GEOAI_MAX_CONCURRENT_JOBS` y reinicio automático del contenedor |
| `.gitignore` | También se versionan los GeoTIFF de `resultados_referencia/` (mapa de incertidumbre y de zonas de la muestra) |
| `README_PROFESOR.md` y `METODOLOGIA.md` | Endpoints nuevos, sección de transición y resultados en la muestra |

## 3. Pruebas nuevas

Se suman 15 pruebas a las 56 de la v1.2, para un total de 71.

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

## 6. Cómo verificar este informe

- Pruebas: `python -m pytest -v` debe dar 71 aprobadas (reporte en `resultados_referencia/pruebas_pytest.txt`).
- Muestra: `python experiments/analisis_completo.py sample_data/palma_muestra.tif salida --cultivo palma` reproduce `resultados_referencia/muestra/`.

## 7. Pendientes

- Repetir los experimentos con el ortomosaico completo usando la v1.3, y medir el tiempo y la memoria de un análisis completo por la API.
- Decidir el número de zonas (los índices indican k = 3) antes de enviar los formatos de validación al experto.
- Al evaluar la exactitud temática, reportarla por separado para puntos seguros y de transición.
- Probar la imagen Docker.
