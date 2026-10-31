# Metodología del Motor GeoAI Analítico M2M (v1.3)

Este documento explica cómo se construye la zonificación, en qué unidad de análisis se trabaja, qué validez tiene el resultado y cómo se explica cada decisión del modelo. Está escrito para lectores de ciencia de datos y de cartografía.

---

## 1. Datos de entrada

| Aspecto | Valor |
|---|---|
| Sensor | DJI Mavic 3 Multispectral: Green 560 ± 16 nm, Red 650 ± 16 nm, Red Edge 730 ± 16 nm, NIR 860 ± 26 nm. Sin banda azul. Cámara RGB aparte de 20 MP (DJI, s. f.). |
| Procesamiento previo | Pix4Dfields: ortomosaico de reflectancia en GeoTIFF de 5 bandas float32 (Green, Red, Red edge, NIR, Alpha); nodata −10000. |
| Sistema de referencia | EPSG:32618 (WGS 84 / UTM zona 18N), en metros. |
| Caso de estudio | Lote de palma de aceite (finca PALMA, Meta, abril de 2025): 12 044 × 7 539 píxeles de 6.07 cm; 24.1 ha con datos. |

La banda Alpha no es espectral: es una máscara de datos válidos (GDAL, s. f.). Una versión anterior del sistema la leía como infrarrojo; la v1.2 identifica cada banda por su descripción y una prueba de regresión impide que el error vuelva.

## 2. Del ortomosaico al mapa de zonas

1. Ráster multibanda georreferenciado, resolución espacial (GSD) de 6 cm.
2. Álgebra de mapas local: NDVI, NDRE y GNDVI se calculan celda por celda.
   NDVI = (NIR − R)/(NIR + R); NDRE = (NIR − RE)/(NIR + RE); GNDVI = (NIR − G)/(NIR + G).
3. Máscara: se excluyen las celdas sin dato (Alpha y nodata) y las que no son vegetación (NDVI < 0.1).
4. De ráster a tabla: cada celda válida se vuelve una fila de la matriz de características.
5. Estandarización de cada índice (puntaje z).
6. Clasificación no supervisada por píxel con Fuzzy C-Means.
7. De tabla a ráster temático: la etiqueta de zona vuelve a su fila y columna.
8. Generalización cartográfica con una unidad mínima de mapeo de 100 píxeles (≈ 0.37 m²).
9. Vectorización y simplificación de líneas (Douglas-Peucker, tolerancia de 0.2 m).
10. Capa vectorial con tabla de atributos (zona, nombre, área del polígono, área de la zona, NDVI, NDRE, GNDVI, desviación).

En paralelo se calculan los rangos de condición del experto, la transición entre zonas, los patrones DBSCAN, el análisis por palma, la explicabilidad, el mapa de incertidumbre y los puntos de verificación.

## 3. La matriz de características

El agrupamiento opera sobre una tabla: cada fila es un píxel (observación) y cada columna un atributo. En PALMA la matriz tiene unos 65.5 millones de filas y 3 atributos. Las coordenadas se conservan solo para devolver el resultado al mapa: el agrupamiento ocurre en el espacio de atributos (espectral), no en el espacio geográfico.

Ejemplo con píxeles reales de PALMA (u = grado de pertenencia a cada zona):

| fila | col | x (UTM) | y (UTM) | NDVI | NDRE | GNDVI | u zona 2 | u zona 1 | u zona 0 | u zona 3 | zona |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4067 | 7807 | 677962.0 | 442469.0 | 0.883 | 0.371 | 0.783 | 0.99 | 0.01 | 0.00 | 0.00 | 2 |
| 3772 | 7813 | 677962.3 | 442486.9 | 0.855 | 0.306 | 0.744 | 0.00 | 0.99 | 0.00 | 0.00 | 1 |
| 3062 | 10759 | 678141.1 | 442530.0 | 0.808 | 0.279 | 0.685 | 0.01 | 0.06 | 0.93 | 0.00 | 0 |
| 5916 | 11795 | 678204.0 | 442356.8 | 0.269 | 0.096 | 0.333 | 0.00 | 0.00 | 0.01 | 0.99 | 3 |
| 3711 | 4967 | 677789.6 | 442490.6 | 0.897 | 0.334 | 0.774 | 0.50 | 0.44 | 0.06 | 0.00 | 2 |

La última fila es un píxel de transición: pertenece en 50 % a la zona 2 y en 44 % a la zona 1. Su explicación local: NDVI aporta +0.145 y GNDVI +0.126 a favor de la zona 2, y NDRE −0.213 en contra; el total, +0.058, decide la zona 2. La tabla se reproduce con `experiments/explicabilidad.py`.

## 4. Unidad de análisis: píxel, celda u objeto

| Unidad | Una fila es | Uso en el proyecto |
|---|---|---|
| Píxel (6 cm) | Una celda del ortomosaico | Zonificación FCM y rangos de NDVI |
| Celda agregada (0.5–2 m) | El promedio de un bloque de píxeles | Análisis de sensibilidad (MAUP) |
| Objeto | Una copa de palma detectada | Conteo y condición por palma (OBIA) |

Con píxeles mucho más pequeños que los objetos (6 cm frente a copas de 8–10 m), la clasificación por píxel capta la variación dentro del objeto (hoja, sombra, suelo): es la resolución H (Strahler et al., 1986) y produce el efecto "sal y pimienta". La respuesta estándar es el análisis basado en objetos (Blaschke, 2010), que con imágenes de dron mejora la clasificación frente al enfoque por píxel (Deng et al., 2024). El resultado también depende de la unidad elegida (MAUP; Mîț et al., 2022), por lo que se reporta la sensibilidad a 0.5, 1 y 2 m. El suavizado espacial posterior al agrupamiento es una práctica establecida en la delimitación de zonas de manejo (Javadi et al., 2022).

## 5. Algoritmos

### 5.1 K-means
Minimiza J = Σᵢ Σₖ uᵢₖ ‖xᵢ − cₖ‖² con uᵢₖ ∈ {0, 1}: cada píxel pertenece a un solo grupo (Lloyd, 1982).

### 5.2 Fuzzy C-Means (FCM)
Minimiza Jₘ = Σᵢ Σₖ uᵢₖᵐ ‖xᵢ − cₖ‖² con uᵢₖ ∈ [0, 1] y Σₖ uᵢₖ = 1 (Dunn, 1973; Bezdek, 1981).
- Pertenencia: uᵢₖ = 1 / Σⱼ (dᵢₖ / dᵢⱼ)^(2/(m−1)).
- Centroide: cₖ = Σᵢ uᵢₖᵐ xᵢ / Σᵢ uᵢₖᵐ.
- Parámetros: m = 2, k = 4 por defecto, hasta 50 iteraciones, tolerancia 10⁻⁴, entrenamiento sobre 100 000 píxeles al azar con semilla fija 42.

Con m → 1, FCM equivale a K-means. Se usa FCM porque la condición del cultivo cambia de forma gradual; es el método más usado para delimitar zonas de manejo (Fridgen et al., 2004; Heidari et al., 2026).

Si dos centroides iniciales resultan ser el mismo valor (imágenes con muchos píxeles idénticos), FCM no los podría separar; en ese caso el arranque se toma entre valores distintos de la muestra. Con datos reales esto no ocurre y el resultado no cambia.

### 5.3 Línea base (modelo de referencia)
Se guardan los centroides, el escalador, el umbral de NDVI, el percentil 99 de las distancias y las etiquetas del experto. Los vuelos siguientes se asignan al centroide congelado más cercano, la regla de asignación de K-means, sin reentrenar, y usan el mismo k y el mismo umbral de NDVI de la línea base para que la vegetación analizada sea comparable. Un reajuste de FCM iniciado desde la línea base se usa solo para medir el desplazamiento de los centroides. El cambio se reporta por número de zona.

### 5.4 DBSCAN
Agrupa por coordenadas en metros (radio de 0.5 m, mínimo 30 puntos) el 5 % de píxeles más alejados de su centroide, para encontrar parches espaciales atípicos. Si ese 5 % supera 200 000 píxeles, se conservan los 200 000 más alejados en lugar de una muestra al azar, para no romper la densidad de los parches. Cada patrón se entrega como un polígono.

### 5.5 Transición entre zonas
El mapa de zonas asigna cada píxel a la zona de mayor pertenencia, que es la práctica estándar para un mapa operativo. El grado de pertenencia no se descarta: un píxel cuya pertenencia máxima es menor a 0.6 se considera de transición, y su segunda zona más cercana indica con cuál se mezcla.

- Reporte por zona: porcentaje y área de la zona que está en transición, y con qué otras zonas se mezclan esos píxeles.
- Mapa de zonas con transición: GeoTIFF con el número de zona en cada píxel y el valor 200 para la transición.
- Las zonas no cambian: un píxel de transición conserva su zona, de modo que las áreas, la línea base y los puntos de verificación son los mismos con o sin esta capa.

No debe confundirse con el rango "transición" de la escala del codirector (NDVI entre 0.5 y 0.6): ese rango describe la condición del cultivo según su NDVI, mientras que un píxel de transición entre zonas es uno cuya zona FCM es poco segura, sea cual sea su NDVI.

La transición no se vectoriza. A 6 cm por píxel está casi toda en los bordes de las copas: en la muestra, entre el 85 % y el 97 % de los píxeles de transición forman parches de menos de 1 m², que como polígonos serían cientos de fragmentos sin uso en campo. Por eso se entrega como estadística y como ráster.

### 5.6 Análisis por palma
- Copas detectadas en el infrarrojo cercano, sobre una malla de 0.30 m, con diferencia de gaussianas (σ = 0.11 y 0.39 veces la distancia de siembra) y supresión de no máximos.
- Las detecciones de bajo contraste solo se aceptan si tienen al menos 3 palmas fuertes a la distancia de siembra (malla triangular).
- NDVI y NDRE por copa en un disco de 0.25 veces la distancia de siembra.
- A inspeccionar: NDVI de copa ≤ 0.5 (criterio del codirector) o al menos 0.10 por debajo de la mediana de sus vecinas en 30 m.
- Posible faltante: espacio de la malla con al menos 5 vecinas a la distancia de siembra y sin copa detectada.

## 6. Criterio experto

Escala de NDVI del codirector (Acta N.° 01, 17/10/2026): mayor a 0.6, buena condición; 0.5 a 0.6, transición; 0.2 a 0.5, estrés y zona de inspección; menor a 0.2, suelo o senescencia. Un NDVI bajo no implica daño: puede ser madurez o senescencia. El sistema reporta área y porcentaje por rango, y la decisión de intervenir corresponde al productor.

## 7. Validez

| Tipo | Pregunta | Método |
|---|---|---|
| Interna | ¿Los grupos están separados y compactos? | Silhouette (Rousseeuw, 1987), Davies-Bouldin (1979), coeficiente de partición FPC (Bezdek, 1981) |
| Número de zonas | ¿Cuántas zonas tiene el lote? | Mínimos del FPI y de la entropía de partición normalizada (Odeh et al., 1992; Fridgen et al., 2004; Moharana et al., 2020; Jovanović et al., 2025) |
| Exactitud temática | ¿El mapa coincide con el terreno? | Muestreo aleatorio estratificado por zona (30 puntos por zona, distancia mínima de 10 m) y matriz de confusión con exactitud global, del usuario, del productor y Kappa (Congalton, 1991; Olofsson et al., 2014) |
| Consistencia con el criterio experto | ¿El mapa respeta la escala de NDVI del codirector? | La escala aplicada a la reflectancia de cada punto y polígono, frente al rango de su zona. Es una verificación interna: usa el mismo NDVI que el modelo y no sustituye la exactitud temática |
| Estabilidad de la línea base | ¿La línea base inventa cambios? | Experimento controlado con estrés simulado en la mitad del lote |
| Escala | ¿El resultado depende de la unidad? | Sensibilidad a 0.5, 1 y 2 m |

## 8. Explicabilidad

1. Prototipos. FCM es un modelo basado en prototipos: cada zona se describe por su centroide en unidades originales.
2. Importancia por permutación (Breiman, 2001): porcentaje de píxeles que cambian de zona al desordenar un índice.
3. Explicación local exacta y aditiva. Para un píxel asignado a la zona A cuya segunda opción es B, la ventaja d_B² − d_A² se descompone exactamente como Σ_f [(z_f − c_Bf)² − (z_f − c_Af)²]. Los aportes suman la decisión, como ocurre con los valores SHAP (Lundberg & Lee, 2017; Lundberg et al., 2020).
4. Modelo sustituto. Un árbol de decisión de profundidad 3 imita al FCM y da reglas legibles, con su fidelidad medida en datos no usados para entrenarlo. Con índices muy correlacionados (NDVI–GNDVI ≈ 0.95), la importancia se reparte entre ellos y debe leerse por grupos.
5. Incertidumbre. El grado de pertenencia de cada píxel a su zona se entrega como GeoTIFF en porcentaje; valores bajos indican transiciones entre zonas. El reporte de transición (sección 5.5) resume esa incertidumbre por zona.

## 9. Resultados en el lote de palma

Los archivos completos están en `resultados_referencia/` y se reproducen con los scripts de `experiments/`. Los resultados del lote completo se obtuvieron con la v1.2; la v1.3 asigna cada píxel a la misma zona, y las áreas solo cambian desde el cuarto decimal porque ahora se calculan con la resolución sin redondear.

| Aspecto | Resultado |
|---|---|
| Condición por rangos | 96.85 % del área con NDVI > 0.6; 0.75 % en transición; 2.24 % en estrés (0.54 ha, 14 zonas de inspección); 0.16 % suelo o senescencia |
| Zonas FCM (k = 4) | Prototipos: NDVI 0.886 / 0.856 / 0.792 / 0.283 y NDRE 0.369 / 0.310 / 0.268 / 0.118. Silhouette 0.345, Davies-Bouldin 0.812, FPC 0.624 |
| Número de zonas | FPI y entropía mínimos en k = 3 (0.400 y 0.407); el lote de plátano coincide (0.371 y 0.406) |
| Línea base | 0.0 % de cambio falso en la mitad sin cambio frente a 34.2 % al reajustar los centroides; 64.9 % de la mitad estresada baja de zona |
| Importancia por permutación | NDRE 42.8 %, GNDVI 28.0 %, NDVI 16.6 % |
| Árbol sustituto | Fidelidad del 89.9 %, con reglas basadas en NDRE y GNDVI |
| Incertidumbre | Pertenencia máxima media 0.74; 23.8 % de píxeles de transición |
| Transición por zona (muestra de 90 × 90 m, k = 4) | 15.1 % de píxeles en transición: 28.9 % de la zona 0, 23.0 % de la zona 1, 13.2 % de la zona 3 y 1.7 % de la zona de estrés. Las zonas se mezclan en cadena: la 3 con la 1, la 1 con la 0 y la 0 con la 2. Con k = 3 la transición baja a 7.1 % |
| Escala | Palma (k = 4): Silhouette 0.345 por píxel, 0.356 a 1 m y 0.358 a 2 m. Plátano (k = 3): 0.502, 0.601 y 0.619 |
| Por palma | 3 067 palmas, 146.3/ha frente a 142.6 esperadas; 23 a inspeccionar; 363 copas débiles; 161 posibles faltantes |
| Contraste con la escala | 116 de 120 puntos y 36 de 36 polígonos en el rango de su zona; las zonas 0, 1 y 2 quedan las tres en "buena condición" |

### Interpretación

- El índice que más pesa es el NDRE. Las zonas de palma se separan sobre todo por el borde rojo y no por el NDVI. Es coherente con la saturación del NDVI en dosel denso (Mutanga & Skidmore, 2004; Vera-Esmeraldas et al., 2025) y con el criterio del codirector. Tres de las cuatro zonas subdividen el rango de buena condición, y la selección de k indica que tres zonas describen mejor el lote.
- La escala pesa según el tipo de dosel. En la palma, de dosel cerrado, agregar los píxeles casi no cambia la estructura. En el plátano, con más suelo y sombra entre plantas, la agregación mejora claramente la separación. Esto muestra que la unidad de análisis debe elegirse según el cultivo.
- La zona de estrés es la mejor definida. Menos del 2 % de sus píxeles está en transición, frente a más del 20 % en las subzonas de buena condición, que se mezclan entre sí en cadena. Es otra señal de que las subzonas por encima de NDVI 0.6 son un gradiente y no condiciones separadas, y de que tres zonas describen mejor el lote.
- El experto no se puede omitir. El 23.8 % de píxeles de transición confirma que las fronteras entre zonas son graduales, lo que justifica FCM y la validación experta. Además, al aplicar la escala del codirector a la reflectancia, 35 de los 36 polígonos grandes quedan en "buena condición": la regla de umbrales confirma la zona de estrés, pero no separa las zonas 0, 1 y 2 ni distingue palma de pasto o arvenses, que también superan 0.6. Llenar los formatos de validación con esa regla sería comparar el modelo consigo mismo; qué cobertura hay en cada zona solo lo establece el experto.

## 10. Limitaciones y trabajo futuro

- No hay dos vuelos del mismo lote: la línea base se valida con un vuelo simulado.
- La exactitud temática y la pureza por zona dependen de la verificación del experto, pendiente.
- El conteo de palmas no se ha comparado con un conteo en campo; arbustos de linderos y vías internas generan falsos positivos conocidos.
- Los porcentajes se calculan sobre el área con datos de la imagen; falta recortar con el polígono del lote.
- Sin banda azul no se calcula EVI; el EVI2 (Jiang et al., 2008) requiere confirmar la escala de reflectancia. Sin infrarrojo de onda corta, el estrés hídrico solo se infiere (Gao, 1996).
- Por encima de NDVI 0.6, las subzonas requieren calibración con datos de campo.
- FCM no incorpora la vecindad espacial; las variantes espaciales de FCM (Ahmed et al., 2002) quedan como trabajo futuro, junto con el aprendizaje a partir de la retroalimentación del usuario.

## 11. Glosario

| Término | Significado en el proyecto |
|---|---|
| Observación (registro, instancia) | Un píxel; o una palma en el análisis por objetos |
| Atributo (variable, característica) | NDVI, NDRE o GNDVI |
| Matriz de características | La tabla píxeles × atributos |
| Espacio de atributos | Donde se agrupa; el espacio geográfico es donde se dibuja |
| Prototipo o centroide | Perfil promedio de una zona |
| Grado de pertenencia | Qué tanto pertenece un píxel a una zona (0 a 1) |
| Píxel de transición | Píxel cuya pertenencia máxima es menor a 0.6: queda entre dos zonas |
| Ráster temático | Mapa de zonas antes de vectorizar |
| Unidad mínima de mapeo | Parche más pequeño que se representa |
| Exactitud temática | Coincidencia del mapa con la referencia de campo |
| Matriz de confusión | Tabla de clases del mapa frente a clases de referencia |
| MAUP | Dependencia del resultado respecto de la unidad de análisis |
| OBIA | Análisis basado en objetos |
| Línea base | Modelo de referencia validado contra el que se comparan los vuelos |

## 12. Referencias

Ahmed, M. N., Yamany, S. M., Mohamed, N., Farag, A. A., & Moriarty, T. (2002). A modified fuzzy c-means algorithm for bias field estimation and segmentation of MRI data. *IEEE Transactions on Medical Imaging, 21*(3), 193–199. https://doi.org/10.1109/42.996338

Ali, A., Rondelli, V., Martelli, R., Falsone, G., Lupia, F., & Barbanti, L. (2022). Management zones delineation through clustering techniques based on soils traits, NDVI data, and multiple year crop yields. *Agriculture, 12*(2), 231. https://doi.org/10.3390/agriculture12020231

Bezdek, J. C. (1981). *Pattern recognition with fuzzy objective function algorithms*. Plenum Press.

Blaschke, T. (2010). Object based image analysis for remote sensing. *ISPRS Journal of Photogrammetry and Remote Sensing, 65*(1), 2–16. https://doi.org/10.1016/j.isprsjprs.2009.06.004

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Congalton, R. G. (1991). A review of assessing the accuracy of classifications of remotely sensed data. *Remote Sensing of Environment, 37*(1), 35–46. https://doi.org/10.1016/0034-4257(91)90048-B

Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 1*(2), 224–227. https://doi.org/10.1109/TPAMI.1979.4766909

Deng, H., Zhang, W., Zheng, X., & Zhang, H. (2024). Crop classification combining object-oriented method and random forest model using unmanned aerial vehicle (UAV) multispectral image. *Agriculture, 14*(4), 548. https://doi.org/10.3390/agriculture14040548

DJI. (s. f.). *Mavic 3M: especificaciones*. https://ag.dji.com/mavic-3-m/specs

Dunn, J. C. (1973). A fuzzy relative of the ISODATA process and its use in detecting compact well-separated clusters. *Journal of Cybernetics, 3*(3), 32–57. https://doi.org/10.1080/01969727308546046

Fridgen, J. J., Kitchen, N. R., Sudduth, K. A., Drummond, S. T., Wiebold, W. J., & Fraisse, C. W. (2004). Management Zone Analyst (MZA): Software for subfield management zone delineation. *Agronomy Journal, 96*(1), 100–108. https://doi.org/10.2134/agronj2004.1000

Gao, B.-C. (1996). NDWI—A normalized difference water index for remote sensing of vegetation liquid water from space. *Remote Sensing of Environment, 58*(3), 257–266. https://doi.org/10.1016/S0034-4257(96)00067-3

GDAL. (s. f.). *RFC 15: Band masks*. https://gdal.org/en/stable/development/rfc/rfc15_nodatabitmask.html

Gitelson, A., & Merzlyak, M. N. (1994). Spectral reflectance changes associated with autumn senescence of *Aesculus hippocastanum* L. and *Acer platanoides* L. leaves. *Journal of Plant Physiology, 143*(3), 286–292. https://doi.org/10.1016/S0176-1617(11)81633-0

Gitelson, A. A., Kaufman, Y. J., & Merzlyak, M. N. (1996). Use of a green channel in remote sensing of global vegetation from EOS-MODIS. *Remote Sensing of Environment, 58*(3), 289–298. https://doi.org/10.1016/S0034-4257(96)00072-7

Heidari, R., Khanmohammadi, R., & Samavati, F. F. (2026). The evolution of data-driven management zone delineation: A systematic review. *Sensors, 26*(10), 3249. https://doi.org/10.3390/s26103249

Javadi, S. H., Guerrero, A., & Mouazen, A. M. (2022). Clustering and smoothing pipeline for management zone delineation using proximal and remote sensing. *Sensors, 22*(2), 645. https://doi.org/10.3390/s22020645

Jiang, Z., Huete, A. R., Didan, K., & Miura, T. (2008). Development of a two-band enhanced vegetation index without a blue band. *Remote Sensing of Environment, 112*(10), 3833–3845. https://doi.org/10.1016/j.rse.2008.06.006

Jovanović, D., Govedarica, M., Gavrilović, M., Čabilovski, R., & van der Wal, T. (2025). Multi-source data integration for sustainable management zone delineation in precision agriculture. *Sustainability, 17*(15), 6931. https://doi.org/10.3390/su17156931

Kurihara, J., Koo, V.-C., Guey, C. W., Lee, Y. P., & Abidin, H. (2022). Early detection of basal stem rot disease in oil palm tree using unmanned aerial vehicle-based hyperspectral imaging. *Remote Sensing, 14*(3), 799. https://doi.org/10.3390/rs14030799

Lloyd, S. (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory, 28*(2), 129–137. https://doi.org/10.1109/TIT.1982.1056489

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. En *Advances in Neural Information Processing Systems 30*. https://papers.nips.cc/paper/7062-a

Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., & Lee, S.-I. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence, 2*(1), 56–67. https://doi.org/10.1038/s42256-019-0138-9

Mîț, A., Reiche, J., Verbesselt, J., & Herold, M. (2022). A downsampling method addressing the modifiable areal unit problem in remote sensing. *Remote Sensing, 14*(21), 5538. https://doi.org/10.3390/rs14215538

Moharana, P. C., Jena, R. K., Pradhan, U. K., Nogiya, M., Tailor, B. L., Singh, R. S., & Singh, S. K. (2020). Geostatistical and fuzzy clustering approach for delineation of site-specific management zones and yield-limiting factors in irrigated hot arid environment of India. *Precision Agriculture, 21*(2), 426–448. https://doi.org/10.1007/s11119-019-09671-9

Mutanga, O., & Skidmore, A. K. (2004). Narrow band vegetation indices overcome the saturation problem in biomass estimation. *International Journal of Remote Sensing, 25*(19), 3999–4014. https://doi.org/10.1080/01431160310001654923

Odeh, I. O. A., McBratney, A. B., & Chittleborough, D. J. (1992). Soil pattern recognition with fuzzy-c-means: Application to classification and soil-landform interrelationships. *Soil Science Society of America Journal, 56*(2), 505–516. https://doi.org/10.2136/sssaj1992.03615995005600020027x

Olofsson, P., Foody, G. M., Herold, M., Stehman, S. V., Woodcock, C. E., & Wulder, M. A. (2014). Good practices for estimating area and assessing accuracy of land change. *Remote Sensing of Environment, 148*, 42–57. https://doi.org/10.1016/j.rse.2014.02.015

Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics, 20*, 53–65. https://doi.org/10.1016/0377-0427(87)90125-7

Strahler, A. H., Woodcock, C. E., & Smith, J. A. (1986). On the nature of models in remote sensing. *Remote Sensing of Environment, 20*(2), 121–139. https://doi.org/10.1016/0034-4257(86)90018-0

Thapa, A., Horanont, T., Neupane, B., Klaylee, J., & Witayangkurn, A. (2025). Classification of oil palm tree conditions from UAV imagery using the YOLO object detector. *Big Earth Data, 9*(4), 860–886. https://doi.org/10.1080/20964471.2025.2491881

Vera-Esmeraldas, A., Pizarro-Otéiza, S., Labbé, M., Rojo, F., & Salazar, F. (2025). UAV-based spectral and thermal indices in precision viticulture: A review of NDVI, NDRE, SAVI, GNDVI, and CWSI. *Agronomy, 15*(11), 2569. https://doi.org/10.3390/agronomy15112569

Yarak, K., Witayangkurn, A., Kritiyutanont, K., Arunplod, C., & Shibasaki, R. (2021). Oil palm tree detection and health classification on high-resolution imagery using deep learning. *Agriculture, 11*(2), 183. https://doi.org/10.3390/agriculture11020183
