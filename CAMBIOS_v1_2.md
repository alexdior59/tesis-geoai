# Cambios y pruebas de la versión 1.2 del Motor GeoAI Analítico M2M

## 1. Propósito y base de comparación

Este informe explica cada cambio hecho al código, el fundamento de cada uno, qué verifica cada prueba automática y qué contiene la entrega, archivo por archivo. La base de comparación es la versión 2 de Julián Díaz (FCM + DBSCAN con un primer modelo de referencia). El detalle línea por línea está en `cambios_frente_a_version_julian.patch`, en la raíz de la entrega.

Historia resumida del código:

| Versión | Contenido | Problema encontrado |
|---|---|---|
| Inicial | K-means + DBSCAN; NDVI, SAVI y GNDVI | Leía las bandas por posición fija: tomaba la banda Alpha como infrarrojo, así que todos los índices eran incorrectos |
| 1 (Julián) | Cambio a Fuzzy C-Means; DBSCAN sobre coordenadas en metros; filtro de parches en el GeoJSON | Seguía el error de bandas |
| 2 (Julián) | Bandas identificadas por nombre y máscara Alpha; primer modelo de referencia y reporte de cambio | Los centroides de referencia se reajustaban en cada vuelo (34,2 % de cambio falso); referencias en memoria; normalización banda por banda |
| 1.2 (esta entrega) | Línea base congelada y persistente; criterio del experto; análisis por palma; explicabilidad; incertidumbre; exactitud temática; 56 pruebas | Ver limitaciones (sección 7) |

## 2. Cambios en el código y su fundamento

### 2.1 app/models/raster_processor.py (modificado)

| Cambio | Por qué | Fundamento |
|---|---|---|
| Un solo factor de escala para las cuatro bandas, decidido con el máximo global, en vez de decidir banda por banda | En la versión 2, si una sola banda superaba 1,0 se dividía solo esa banda y los índices quedaban distorsionados | Los índices de diferencia normalizada (NDVI, NDRE, GNDVI) no cambian si todas las bandas se multiplican por el mismo factor, pero sí si cada banda tiene un factor distinto |
| Se conserva la detección de bandas por nombre y la máscara Alpha de la versión 2 | Los GeoTIFF de Pix4Dfields para el Mavic 3M traen Green, Red, Red edge, NIR y Alpha, sin banda azul | Especificaciones de DJI; la banda Alpha es una máscara de datos válidos (GDAL RFC 15) |

### 2.2 app/models/spatial_segmenter.py (modificado)

| Cambio | Por qué | Fundamento |
|---|---|---|
| Con línea base, cada píxel se asigna al centroide congelado más cercano; ya no se reentrena FCM | La observación del director: cada análisis creaba clústeres nuevos sin punto fijo de comparación. Con el reajuste, el 34,2 % del área que no cambió aparecía como cambiada | Detección de cambios: comparar contra un modelo de referencia fijo (regla de asignación de K-means). Experimento controlado: 0,0 % de cambio falso |
| El reajuste desde la línea base se usa solo para medir el desplazamiento de los centroides | Se conserva la información de cuánto se movió cada zona sin cambiar la clasificación | Separar la clasificación de la medición del cambio |
| El k debe coincidir con el de la línea base (error explícito) | La versión 2 iniciaba al azar sin avisar, pero seguía usando el escalador de la referencia | Fallar de forma explícita en lugar de dar un resultado sin sentido |
| El cambio por zona se compara por número de zona (cluster_id) y no por posición en una lista ordenada | Si una zona desaparecía, todas las comparaciones quedaban corridas | Con centroides congelados, el número de zona identifica la misma condición en todos los vuelos |
| La desviación se normaliza con el percentil 99 de la línea base | Normalizar con cada imagen ocultaba un empeoramiento general del lote | Una escala fija permite comparar entre vuelos |
| anomaly_score pasa a deviation_score y las zonas se nombran con la escala del experto (buena_condicion, transicion, estres, suelo_o_senescencia) | El codirector pidió no asociar los valores bajos con daño | Acta N.° 01, numerales 3.1 a 3.3 |
| Cada zona incluye el nombre que le dio el experto (expert_label) | Las etiquetas del experto se vuelven parte del modelo | Validación experta como verdad de referencia |
| Nueva métrica: coeficiente de partición difusa (FPC) | El Silhouette mide particiones duras; FCM es difuso | Bezdek (1981) |
| Pertenencia máxima de cada píxel (max_membership) | Base del mapa de incertidumbre | Fórmula de pertenencia de FCM (Bezdek, 1981) |
| Separación de _fit_fcm (aprender centroides) y _assign (asignar en lotes) | Permite clasificar con centroides congelados y reusar la asignación | Mismo muestreo y semilla: con datos iguales da exactamente las zonas de la versión 2 |

### 2.3 app/models/crop_profiles.py (nuevo)

Contiene la escala de NDVI del codirector (mayor a 0,6 buena condición; 0,5–0,6 transición; 0,2–0,5 estrés; menor a 0,2 suelo o senescencia), el rango de inspección, los parámetros de palma (siembra a 9 m, alerta de copa en 0,5) y la fuente de cada umbral. Fundamento: Acta N.° 01; los rangos dependen del cultivo, por eso cada perfil exige una fuente.

### 2.4 app/models/condition_report.py (nuevo)

| Función | Fundamento |
|---|---|
| Área y porcentaje por rango de NDVI sobre todos los píxeles con dato | Acuerdo 4.2: reportar porcentaje y área; la decisión de intervenir es del productor |
| Zonas de inspección: píxeles con NDVI entre 0,2 y 0,5, generalizados con una unidad mínima de mapeo de 10 m², separados en componentes conectados y vectorizados | Unidad mínima de mapeo (generalización cartográfica); zonas útiles para ir a campo |
| Coordenadas de cada zona en UTM y en latitud/longitud WGS84 | Llegar al punto con el GPS del celular |
| Nota de lenguaje neutral en el resultado | Acta N.° 01, numeral 3.3 |

### 2.5 app/models/palm_analyzer.py (nuevo)

| Paso | Cómo | Fundamento |
|---|---|---|
| Detección de copas | Banda NIR promediada a 0,30 m; diferencia de gaussianas con σ = 0,11 y 0,39 veces la distancia de siembra; máximos locales | En NIR las copas son manchas brillantes; detección de manchas a la escala del objeto. Análisis basado en objetos (Blaschke, 2010) |
| Supresión de no máximos | De los picos más cercanos que media distancia de siembra se conserva el más fuerte | Error encontrado por las pruebas: una copa centrada entre dos celdas se contaba dos veces |
| Filtro de malla | Las copas de bajo contraste se aceptan solo con al menos 3 palmas fuertes a la distancia de siembra | Distingue palmas jóvenes o enfermas de parches de pasto o arbustos de lindero |
| NDVI y NDRE por copa | Promedio en un disco de 0,25 veces la distancia de siembra | La palma es la unidad de manejo (las labores se hacen por palma) |
| Palmas a inspeccionar | NDVI de copa ≤ 0,5, o al menos 0,10 por debajo de la mediana de las vecinas en 30 m | Criterio del codirector; comparación con las vecinas usando mediana y MAD, que son robustas |
| Posibles faltantes | Máximos de la distancia a la palma más cercana (≥ 0,8 veces la siembra) con al menos 5 vecinas a la distancia de siembra y rodeados en todas las direcciones | En malla triangular una palma faltante tiene 6 vecinas; en una vía solo quedan unas 4 |
| Densidad esperada | 2 / (√3 · s²) palmas por m²; 142,6 palmas/ha a 9 m | Geometría de la siembra en tres bolillo |

### 2.6 app/models/explainability.py (nuevo)

| Pieza | Qué entrega | Fundamento |
|---|---|---|
| Prototipos | Centroide de cada zona en NDVI, NDRE y GNDVI originales y % de píxeles | FCM es un modelo basado en prototipos: cada zona se explica por su perfil |
| Importancia por permutación | % de píxeles que cambian de zona al desordenar cada índice | Breiman (2001); se mide sobre el propio modelo, sin sustituto |
| Modelo sustituto | Árbol de decisión de profundidad 3, su fidelidad en datos no usados y sus reglas | Reglas legibles para el agrónomo; la fidelidad dice qué tanto se puede confiar en ellas |
| Explicación local exacta | Aporte de cada índice a la ventaja de la zona asignada sobre la segunda: d_B² − d_A² = Σ_f [(z_f − c_Bf)² − (z_f − c_Af)²] | Descomposición aditiva y exacta, con la misma propiedad de precisión local de los valores SHAP (Lundberg & Lee, 2017) |
| Incertidumbre | Pertenencia máxima media y % de píxeles de transición (pertenencia < 0,6) | Las fronteras entre condiciones del cultivo son graduales |

### 2.7 app/models/accuracy.py (nuevo)

| Función | Fundamento |
|---|---|
| Muestreo aleatorio estratificado por zona sobre el mapa generalizado (30 puntos por zona por defecto) | Práctica estándar de evaluación de mapas (Olofsson et al., 2014) |
| Distancia mínima de 10 m entre puntos | Reduce la autocorrelación espacial entre puntos vecinos (Tobler, 1970) |
| Identificadores barajados y GeoJSON sin la zona del modelo | Evaluación a ciegas: el experto no debe saber qué dijo el modelo |
| Matriz de confusión (filas = mapa, columnas = referencia), exactitud global, del usuario, del productor y Kappa | Congalton (1991) |

### 2.8 app/views/response_serializer.py (modificado)

- Nuevas secciones del JSON: condicion_por_rangos, analisis_palmas y explicabilidad. Las listas largas se recortan a 200 elementos; el GeoJSON completo sale por su propio endpoint.
- write_uncertainty_tif: GeoTIFF de una banda con la pertenencia en porcentaje (255 = sin dato).
- El GeoJSON de zonas incluye expert_label y deviation_score_mean. El resumen en texto agrega los rangos, el análisis por palma, la explicabilidad y el cambio por zona.
- Descargo de responsabilidad reescrito con lenguaje neutral. Versión de la API: 1.2.0.

### 2.9 app/controllers/analyze_router.py (modificado)

| Cambio | Por qué |
|---|---|
| Parámetro crop_type con validación (422 si el cultivo no existe) | Acuerdo 4.1: el usuario indica el cultivo |
| Líneas base guardadas como JSON en GEOAI_REF_DIR y recargadas al iniciar | En la versión 2 se perdían al reiniciar el servidor |
| Con reference_model_id: 404 si no existe, 422 si es de otro cultivo y k tomado de la línea base | Evitar comparaciones inválidas |
| POST /reference recibe zone_labels (validados contra las zonas del análisis) y validated_by | Guardar el criterio del experto en el modelo |
| Nuevos endpoints: /crops, /result/{id}/inspection, /palms, /uncertainty, /accuracy-sample y POST /accuracy | Exponer cada producto nuevo |
| Resultados en disco en GEOAI_RESULTS_DIR (mapa de incertidumbre) | Los GeoTIFF no caben en la memoria del job |
| El estado 'queued' se trata como 'processing' | Antes devolvía un error confuso si se consultaba muy pronto |

### 2.10 Otros archivos

| Archivo | Cambio | Por qué |
|---|---|---|
| app/main.py | Descripción y versión 1.2.0 | Documentación de la API en /docs |
| requirements.txt | Se agregan scipy y shapely con versión fija | El código las usa directamente; antes llegaban solo como dependencias de otras librerías |
| requirements-dev.txt (nuevo) | pytest y openpyxl | Correr las pruebas y los experimentos con Excel |
| pytest.ini (nuevo) | Ruta del proyecto y carpeta de pruebas | Que python -m pytest funcione desde la raíz |
| docker-compose.yml | Volumen data/references y variables GEOAI_REF_DIR y GEOAI_RESULTS_DIR | Que las líneas base y los resultados persistan fuera del contenedor |
| .gitignore / .dockerignore | Se versionan la muestra y los resultados de referencia; la imagen Docker no incluye pruebas ni datos | Reproducibilidad del repositorio e imagen liviana |
| Dockerfile | Sin cambios | No se probó en el equipo de la entrega (sin Docker) |

## 3. Pruebas automáticas

Hay 56 pruebas que corren en unos 40 segundos con `python -m pytest -v` (reporte en `resultados_referencia/pruebas_pytest.txt`). Principios con los que se diseñaron:

- Imágenes sintéticas con respuesta conocida (oráculo): se generan con la misma estructura de Pix4Dfields (5 bandas float32, nodata −10000, EPSG:32618) y valores de NDVI, NDRE y GNDVI calculados a mano, así el resultado correcto se conoce de antemano.
- Prueba de regresión del error de bandas, para que no vuelva a aparecer.
- Propiedades matemáticas: aditividad exacta, invariancia de escala y rangos teóricos (la pertenencia está entre 1/k y 1).
- Reproducibilidad: misma semilla, mismo resultado.
- Integración por HTTP con el cliente de pruebas de FastAPI, con carpetas temporales aisladas por prueba.
- Sin datos reales: las pruebas son rápidas y corren en cualquier computador. Los datos reales se usan en los experimentos (sección 4).
- Las pruebas encontraron un error real: el detector de palmas contaba dos veces una copa centrada entre dos celdas. Se corrigió con supresión de no máximos.

### 3.1 tests/conftest.py

Generadores de ortomosaicos sintéticos: cuadrantes con NDVI 0,896 / 0,55 / 0,40 / 0,077 (25 % cada uno); tres franjas de condición (vigorosa, buena, estrés) con ruido; un segundo vuelo con el NIR de la mitad izquierda reducido 15 %; y una plantación de 72 × 72 m a 9 m en triángulo (68 posiciones) con una palma faltante y una enferma.

### 3.2 tests/test_raster_processor.py (8)

| Prueba | Qué verifica | Fundamento |
|---|---|---|
| test_mapeo_de_bandas_por_descripcion | Identifica Green=1, Red=2, RedEdge=3, NIR=4, Alpha=5 por nombre | Formato de exportación de Pix4Dfields |
| test_sin_descripciones_usa_orden_del_mavic_3m | Sin nombres de banda usa el orden del Mavic 3M | Respaldo seguro |
| test_regresion_alpha_no_se_usa_como_nir | NDVI de 0,897 y no ~0,3 con la banda Alpha en 255 | Regresión del error original |
| test_indices_con_valores_conocidos | NDVI, NDRE y GNDVI iguales a la fórmula | Oráculo analítico |
| test_mascara_alpha_y_umbral_de_vegetacion | Alpha = 0 queda sin dato; NDVI < 0,1 sale del agrupamiento pero cuenta en los rangos | Máscara de datos y de vegetación |
| test_un_solo_factor_de_escala_preserva_los_indices | Multiplicar todas las bandas por 10 000 no cambia los índices | Invariancia de escala |
| test_banda_mayor_a_uno_no_distorsiona_el_ndvi | Una banda mayor a 1 no distorsiona el NDVI | Corrige la normalización banda por banda |
| test_menos_de_cuatro_bandas_es_error | Rechaza archivos con menos de 4 bandas | Validación de entrada |

### 3.3 tests/test_crop_profiles.py (9)

| Prueba | Qué verifica | Fundamento |
|---|---|---|
| test_rangos_contiguos_sin_huecos | Los rangos cubren de −1 a 1 sin huecos ni traslapes | Toda área queda en un rango |
| test_escala_del_codirector (6 casos) | 0,05 es suelo; 0,2 y 0,45, estrés; 0,55, transición; 0,6 y 1,0, buena condición | Valores límite de la escala del acta |
| test_perfil_palma_trae_parametros_de_siembra | Siembra 9 m, alerta 0,5, rango 0,2–0,5 y fuente citada | Acta N.° 01 |
| test_cultivo_no_soportado | Cultivo inexistente da error | Validación de entrada |

### 3.4 tests/test_condition_report.py (4)

| Prueba | Qué verifica | Fundamento |
|---|---|---|
| test_porcentaje_por_rango_de_ndvi | 25 % exacto en cada rango y el área suma 0,04 ha | Oráculo de cuadrantes |
| test_zona_de_inspeccion_georreferenciada | Una zona de 100 m² con NDVI 0,40 y coordenadas en el Meta | Georreferenciación correcta |
| test_parches_menores_al_minimo_no_son_zonas | Un parche de 4 m² cuenta en el área pero no es zona | Unidad mínima de mapeo de 10 m² |
| test_lenguaje_neutral | Nota sin connotación de daño y fuente del codirector | Acta N.° 01 |

### 3.5 tests/test_spatial_segmenter.py (9)

| Prueba | Qué verifica | Fundamento |
|---|---|---|
| test_fcm_recupera_las_tres_condiciones | Pureza > 99 % de cada franja | El algoritmo encuentra condiciones conocidas |
| test_resultado_reproducible | Dos corridas dan zonas idénticas | Semilla fija |
| test_metricas_de_calidad_en_rango | FPC entre 1/k y 1; Silhouette > 0,5 | Rangos teóricos de Bezdek y Rousseeuw |
| test_zonas_ordenadas_y_con_nombre_de_la_escala_del_experto | Zonas ordenadas por NDVI y nombradas con la escala; ninguna 'anomalía' | Lenguaje del acta |
| test_linea_base_mismo_vuelo_no_reporta_cambio | Mismo vuelo contra su línea base: etiquetas idénticas y cambio cero | Estabilidad de la línea base |
| test_linea_base_congelada_detecta_estres_sin_cambios_falsos | Mitad sin cambio: 0 % de cambio; mitad estresada: > 5 % | Respuesta a la observación del director |
| test_cambio_reportado_por_zona_con_etiquetas_del_experto | Cambio por número de zona con el nombre del experto y su validador | Trazabilidad del criterio experto |
| test_k_distinto_al_de_la_linea_base_es_error | k distinto al de la línea base da error | Evitar comparaciones inválidas |
| test_desviacion_normalizada_con_la_linea_base | Usa el percentil 99 de la línea base; desviación entre 0 y 1 | Escala fija entre vuelos |

### 3.6 tests/test_palm_analyzer.py (7)

| Prueba | Qué verifica | Fundamento |
|---|---|---|
| test_conteo_de_palmas | Cuenta las 66 palmas sanas (± 1) | Oráculo de plantación sintética |
| test_densidad_coherente_con_siembra_a_9_m | Densidad esperada de 142,6/ha y medida dentro del 15 % | Geometría de la siembra |
| test_cada_palma_sana_se_detecta_una_vez | Ninguna palma duplicada ni omitida | Supresión de no máximos |
| test_palma_faltante_se_reporta | La palma quitada aparece como posible faltante a menos de 2,5 m | Detección de huecos en la malla |
| test_palma_enferma_queda_para_revision | La palma pálida aparece para inspección, como copa débil o como faltante | Error típico del conteo con dron (codirector) |
| test_palmas_sanas_no_se_marcan | Como máximo 1 falsa alarma | Control de falsos positivos |
| test_salida_con_coordenadas_gps | Todas las salidas con latitud y longitud válidas | Uso en campo |

### 3.7 tests/test_explainability.py (6)

| Prueba | Qué verifica | Fundamento |
|---|---|---|
| test_pertenencia_maxima_en_rango_y_media_en_la_frontera | Pertenencia 1 en el centroide y 0,5 en la frontera | Fórmula de FCM |
| test_explicacion_local_es_aditiva_y_exacta | Los aportes suman exactamente d_B² − d_A² | Precisión local (propiedad SHAP) |
| test_importancia_identifica_el_indice_informativo | Si solo NDRE separa las zonas: NDRE > 30 %, los otros < 5 % | Validez de la importancia por permutación |
| test_modelo_sustituto_fiel_y_con_reglas | Fidelidad > 95 % y reglas basadas en NDRE | Confiabilidad del sustituto |
| test_prototipos_en_unidades_originales | Centroides en NDRE 0,25 y 0,38; porcentajes suman 100 | Interpretación en unidades reales |
| test_explicabilidad_sobre_la_segmentacion_real | Pertenencia entre 1/3 y 1; 3 ejemplos típicos y 3 de transición | Integración con el segmentador |

### 3.8 tests/test_accuracy.py (6)

| Prueba | Qué verifica | Fundamento |
|---|---|---|
| test_matriz_de_confusion_con_valores_conocidos | Exactitud global 5/6, Kappa 17/23 y exactitudes por clase calculadas a mano | Congalton (1991) |
| test_acuerdo_perfecto_da_kappa_uno | Acuerdo total da exactitud 1 y Kappa 1 | Caso límite |
| test_listas_invalidas | Listas vacías o de distinto largo dan error | Validación |
| test_muestreo_estratificado_por_zona | 5 puntos por zona, distancia mínima respetada, identificadores 1 a 15 | Muestreo estratificado |
| test_muestreo_reproducible | Misma semilla, mismos puntos | Reproducibilidad |
| test_geojson_a_ciegas_no_revela_la_zona | El GeoJSON para el experto no trae la zona | Evaluación a ciegas |

### 3.9 tests/test_api.py (7)

| Prueba | Qué verifica | Fundamento |
|---|---|---|
| test_health_y_cultivos | Servicio activo y cultivos disponibles | Disponibilidad |
| test_validaciones_de_entrada | Cultivo inválido 422, archivo no TIF 422, línea base inexistente 404 | Errores claros |
| test_analisis_completo_generico | Todas las secciones, ninguna 'anomalía' y todos los endpoints de resultado | Flujo de punta a punta |
| test_analisis_de_palma | Conteo y GeoJSON con palmas normales y faltantes | Flujo de palma |
| test_explicabilidad_e_incertidumbre | Sección de explicabilidad y descarga del GeoTIFF (EPSG:32618, 33–100 %) | Productos nuevos |
| test_exactitud_tematica_por_api | Puntos a ciegas; respuestas perfectas dan exactitud 1 y Kappa 1; punto inexistente 422 | Exactitud temática por la API |
| test_ciclo_de_vida_de_la_linea_base | Etiquetas inválidas 422; creación 201; archivo en disco; recarga tras reinicio; otro cultivo 422; k de la línea base; cambio cero | Persistencia y uso de la línea base |

## 4. Experimentos con datos reales

| Script | Qué mide | Resultado en PALMA | Fundamento |
|---|---|---|---|
| analisis_completo.py | Pipeline completo sin la API, con todas las salidas | Rangos, zonas, palmas, explicabilidad, incertidumbre y 120 puntos de verificación | Reproducibilidad fuera del servidor |
| linea_base_estres.py | Cambio falso y detección con estrés simulado | 0,0 % frente a 34,2 %; 64,9 % de detección | Diseño experimental controlado |
| seleccion_k.py | FPI, entropía, FPC, Silhouette y Davies-Bouldin para k = 2 a 6 | Mínimos en k = 3 (también en plátano) | Odeh et al. (1992); Fridgen et al. (2004) |
| sensibilidad_escala.py | Píxel frente a celdas de 0,5, 1 y 2 m | Palma casi no cambia; plátano mejora con celdas mayores | MAUP (Mîț et al., 2022) |
| explicabilidad.py | Tabla de atributos, prototipos, permutación, explicación local y reglas | NDRE 42,8 %; árbol con 89,9 % de fidelidad | Breiman (2001); Lundberg & Lee (2017) |
| exactitud_tematica.py | Formulario a ciegas y evaluación con matriz de confusión | Formulario de 120 puntos listo; evaluación pendiente del experto | Congalton (1991); Olofsson et al. (2014) |
| contraste_escala_experto.py | La escala de NDVI del codirector aplicada a la reflectancia de cada punto (mediana en 1 × 1 m) y polígono, frente al rango de su zona | 116 de 120 puntos y 36 de 36 polígonos coinciden; las zonas 0, 1 y 2 quedan las tres en buena condición | Acta N.° 01; verificación interna, no exactitud temática |

### 4.1 Por qué la reflectancia no reemplaza al experto

Se evaluó si la escala del codirector, aplicada automáticamente a la imagen, podía llenar los formatos de validación. La conclusión es que no, por dos razones:

- Circularidad: la regla usa el mismo NDVI con el que el modelo arma las zonas. Comparar el mapa con esa regla mide la consistencia del modelo consigo mismo, no su exactitud. Por eso el acuerdo es alto (96,7 % en puntos) y no demuestra nada sobre el terreno.
- Falta de información: 35 de los 36 polígonos quedan en buena condición (NDVI > 0,6). La regla no separa las zonas 0, 1 y 2, que el modelo distingue por NDRE, ni distingue palma de pasto, arvenses o bosque, que también superan 0,6. Esa diferencia solo la establece la interpretación visual del experto o la visita de campo.

Los formatos de validación siguen en blanco para Cristian. El contraste se entrega como análisis aparte (`Contraste_escala_codirector_PALMA_NO_ENVIAR_AL_EXPERTO.xlsx`), porque muestra la zona del modelo en cada punto.

## 5. Contenido de la entrega, archivo por archivo

### 5.1 Raíz

| Archivo | Contenido |
|---|---|
| LEEME.md | Guía de entrada: qué hay y cómo probar en tres pasos |
| cambios_frente_a_version_julian.patch | Diferencias línea por línea frente a la versión 2 de Julián, para integrarlas en GitHub |

### 5.2 codigo/tesis-geoai-main/

| Archivo o carpeta | Estado | Contenido |
|---|---|---|
| app/main.py | Modificado | Aplicación FastAPI |
| app/controllers/analyze_router.py | Modificado | Endpoints, trabajos asíncronos y líneas base persistentes |
| app/models/raster_processor.py | Modificado | Lectura del GeoTIFF e índices |
| app/models/spatial_segmenter.py | Modificado | FCM, DBSCAN, métricas y línea base congelada |
| app/models/crop_profiles.py | Nuevo | Escala de NDVI por cultivo |
| app/models/condition_report.py | Nuevo | Rangos y zonas de inspección |
| app/models/palm_analyzer.py | Nuevo | Análisis por palma |
| app/models/explainability.py | Nuevo | Explicabilidad e incertidumbre |
| app/models/accuracy.py | Nuevo | Exactitud temática |
| app/views/response_serializer.py | Modificado | JSON, GeoJSON, resumen y GeoTIFF de incertidumbre |
| app//__init__.py | Sin cambios | Paquetes de Python |
| tests/ (conftest.py y 8 archivos) | Nuevo | 56 pruebas automáticas |
| experiments/ (7 scripts) | Nuevo | Experimentos reproducibles |
| sample_data/palma_muestra.tif | Nuevo | Recorte de 90 × 90 m de PALMA (28 MB) para probar la API |
| resultados_referencia/ | Nuevo | Reporte de pruebas, salidas de cada experimento y análisis completo de la muestra |
| README_PROFESOR.md | Nuevo | Instalación, pruebas, uso de la API y resultados |
| METODOLOGIA.md | Nuevo | Método, validez, explicabilidad, glosario y referencias |
| CAMBIOS_v1_2.md | Nuevo | Este informe |
| requirements.txt / requirements-dev.txt / pytest.ini | Modificado / Nuevo / Nuevo | Dependencias y configuración de pruebas |
| Dockerfile / docker-compose.yml / .gitignore / .dockerignore | Sin cambios / Modificado / Modificado / Modificado | Contenedor y control de versiones |

### 5.3 proyecto_qgis_palma/

| Archivo | Capa en QGIS | Fundamento del estilo |
|---|---|---|
| PALMA_proyecto_final.qgz | Proyecto con rutas relativas y vista inicial sobre el lote | Abre en cualquier computador |
| capas/palmas.geojson | Palmas por estado | Amarillo normal, naranja copa débil, rojo inspeccionar, cruz cian faltante |
| capas/puntos_verificacion.geojson | Puntos de verificación numerados (apagada) | Sin la zona del modelo, para la evaluación a ciegas |
| capas/poligonos_grandes.geojson | 36 polígonos numerados (apagada) | Mismos números del formato de validación |
| capas/zonas_inspeccion.geojson | Zonas de inspección | Relleno rojo semitransparente |
| capas/zonas_fcm.geojson + zonas_fcm.qml | Zonas FCM (apagada) | Colores por zona con NDVI y área en la leyenda |
| capas/incertidumbre_30cm.tif | Incertidumbre (apagada) | Rojo = transición; verde = asignación segura; promediada a 30 cm para aligerar |
| capas/ortomosaico_color_natural.tif | Color natural | Composición de Pix4D a partir de las bandas verde y roja |
| capas/ortomosaico_falso_color.tif | Falso color infrarrojo (apagada) | NIR-R-G: vegetación sana en tonos rojizos |

### 5.4 documentos/

| Archivo | Contenido | Estado |
|---|---|---|
| Acta_Reunion_01_Codirector_Cristian (docx, pdf) | Conceptos del codirector y acuerdos | Pendiente de firma |
| Acta_Seguimiento_02_Director (docx, pdf) | Avances y temas para decidir con el director | Para la reunión |
| Informe_cambios_y_pruebas_v1_2 (docx, pdf) | Este informe | Listo |
| Validacion_por_poligono_PALMA.xlsx | Validación a ciegas de 36 polígonos; calcula la pureza | En blanco: lo diligencia el experto |
| Verificacion_exactitud_tematica_PALMA.xlsx | 120 puntos para la matriz de confusión | En blanco: lo diligencia el experto |
| clave_puntos_zona_NO_ENVIAR_AL_EXPERTO.csv | Zona del modelo en cada punto | Solo para evaluar |
| Contraste_escala_codirector_PALMA_NO_ENVIAR_AL_EXPERTO.xlsx | La escala del codirector aplicada a la reflectancia de los 120 puntos y los 36 polígonos | Análisis interno; no se envía al experto |
| Palmas_a_inspeccionar_PALMA.xlsx | Palmas marcadas, copas débiles y faltantes con coordenadas | Columna de verificación en campo en blanco |

## 6. Cómo verificar este informe

- Pruebas: desde codigo/tesis-geoai-main, `python -m pytest -v` debe dar 56 aprobadas.
- Cambios: el archivo .patch muestra cada línea modificada frente a la versión 2.
- Resultados: cada experimento se repite con el script correspondiente y el ortomosaico completo; las salidas de referencia están en resultados_referencia/.

## 7. Limitaciones conocidas

- La línea base se validó con un vuelo simulado: no hay dos vuelos del mismo lote.
- La pureza por zona y la exactitud temática esperan la verificación del experto; los formatos están listos y en blanco. No se pueden llenar con la escala de NDVI (sección 4.1).
- El conteo de palmas no se ha comparado con un conteo en campo; hay falsos positivos conocidos en linderos y vías internas.
- Los porcentajes usan el área con datos de la imagen; falta recortar con el polígono del lote.
- Docker no se probó en el equipo de la entrega.
