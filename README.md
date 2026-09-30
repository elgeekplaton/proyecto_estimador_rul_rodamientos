# 🔧 Estimador de RUL de Rodamientos — XJTU-SY

**Autor:** Pablo Zarate
**Entrega:** Proyecto Final M7 — `zarate_pablo_estimador_rul_rodamientos.zip`

Sistema de mantenimiento predictivo que estima la **vida útil remanente (RUL, *Remaining Useful Life*)** de rodamientos a partir de sus señales de vibración, combinando un **Random Forest** para predecir el RUL en minutos y un modelo **ARIMA([1,5],1,0)** para proyectar la evolución de la degradación.

---

## 1. Problema

Los rodamientos son uno de los componentes que más fallan en la maquinaria rotativa (motores, bombas, ventiladores, cajas reductoras). Una falla inesperada provoca paradas no programadas, daños en otros componentes y costos elevados de reparación.

El mantenimiento tradicional es **correctivo** (se repara cuando falla) o **preventivo por calendario** (se cambia la pieza cada cierto tiempo, aunque esté sana). Ambos enfoques desperdician recursos. El **mantenimiento predictivo** busca intervenir justo antes de la falla, y para eso necesita responder una pregunta concreta:

> *Dado el estado actual de vibración del rodamiento, ¿cuántos minutos de operación le quedan antes de fallar?*

## 2. Objetivo

Construir un modelo de *machine learning* que:

1. **Estime el RUL en minutos** de un rodamiento a partir de características extraídas de sus señales de vibración (**Módulo A — Random Forest**).
2. **Proyecte la evolución del desgaste** a corto plazo mediante un indicador de salud (*Health Indicator*) y un modelo de series de tiempo (**Módulo B — ARIMA([1, 5],1,0)**).
3. Pueda **consultarse desde una interfaz web** sencilla (Streamlit), como prueba de concepto de uso en un entorno industrial.

## 3. Datos

### 3.1 Dataset XJTU-SY

Se utilizó el **XJTU-SY Bearing Dataset**, publicado por la Universidad Xi'an Jiaotong y Changxing Sumyoung Technology (Wang et al., 2020). Contiene ensayos acelerados de vida hasta la falla de rodamientos LDK UER204.

| Parámetro | Valor |
|---|---|
| Condición de operación | 1 — 35 Hz (2100 rpm) / 12 kN de carga radial |
| Rodamientos usados | `Bearing1_1`, `Bearing1_2`, `Bearing1_3` |
| Frecuencia de muestreo | 25,6 kHz |
| Canales | Vibración horizontal y vertical |
| Registro | 1 archivo CSV (snapshot) por minuto de operación |

| Rodamiento | Vida total (min) | FPT detectado (min) | Filas post-FPT |
|---|---|---|---|
| 1_1 | 123 | 72 | 52 |
| 1_2 | 161 | 41 | 121 |
| 1_3 | 158 | 63 | 96 |

> ⚠️ Por las reglas de la entrega, **el dataset completo no se incluye en el ZIP**. Puede descargarse desde el repositorio oficial: <https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets>

> Archivo comprimido en formato .zip, utilizado con los datasets crudos de los rodamientos 1_1, 1_2 y 1_3:  <https://drive.google.com/file/d/1CYhylmuhl_XZk8V3u1WLBtxzbNyUrDTv/view?usp=drive_link>

### 3.2 `datos_muestra.csv`

Muestra de **100 filas** ya procesadas (características extraídas), generada por el notebook de entrenamiento:

- **70 filas `train`**: tomadas de los datos usados para entrenar (1_1, 1_3 y el 70 % inicial de 1_2).
- **30 filas `test`**: tomadas del 30 % final de 1_2, **nunca visto por el modelo**.

Columnas: las 12 características que usa el modelo + `bearing`, `t` (minuto), `rul` (RUL real), `rul_capped` (RUL real con tope de 60 min) y `split` (`train` / `test`).

## 4. Metodología y modelo

### 4.1 Extracción de características

Por cada snapshot y cada canal (horizontal `h_` y vertical `v_`) se calculan **21 características**, 42 en total:

- **Dominio del tiempo:** RMS, pico, pico a pico, curtosis, asimetría, factor de cresta, factor de forma y factor de impulso.
- **Dominio de la frecuencia:** centroide espectral y energía logarítmica en 8 bandas de 0 a 12,8 kHz.
- **Análisis de envolvente:** filtro pasabanda 2–10 kHz + transformada de Hilbert; amplitud de los 3 primeros armónicos de la frecuencia de falla de pista externa (**BPFO ≈ 107,9 Hz**) con una tolerancia de ±3 Hz, y su suma.

### 4.2 Health Indicator (HI) y detección del FPT

- Se ajusta un **PCA (8 componentes) solo con el 15 % inicial de la vida** de cada rodamiento (fase sana) y se usa el **error de reconstrucción** como indicador de salud. Es un HI **causal**: solo usa información del pasado.
- Dentro de la fase sana se usa K-Fold para que el error sea comparable con el resto de la vida y evitar un salto artificial en el borde.
- El **FPT (*First Predicting Time*)** es el primer instante en que el HI supera de forma sostenida (8 minutos seguidos) 10 veces el máximo de su fase sana. Marca el inicio de la degradación.

### 4.3 Módulo A — Predicción de RUL con Random Forest

- Solo se usan datos **posteriores al FPT** (antes del FPT el rodamiento está sano y el RUL no es predecible a partir de la vibración).
- El RUL objetivo se limita a **60 minutos** (`RUL_CAP = 60`), una práctica habitual en pronóstico de vida útil.
- **Selección de características:** se eligen 12 características con mayor monotonía respecto al tiempo (Spearman, tomando el peor caso entre rodamientos) y se descartan las redundantes (correlación > 0,95).

Características finales:
`v_band4`, `v_band5`, `v_band2`, `v_band1`, `h_band1`, `h_p2p`, `h_band2`, `v_peak`, `h_band3`, `v_bpfo_h1`, `v_bpfo_sum`, `h_bpfo_h1`

- **Modelo:** `RandomForestRegressor(n_estimators=200, max_depth=5, random_state=0)`.
- **Partición de datos (esquema híbrido):**
  - **Entrenamiento:** 1_1 completo + 1_3 completo + **70 % inicial** de 1_2 (post-FPT, t = 41 a 124).
  - **Test:** **30 % final** de 1_2 (t = 125 a 161, 37 puntos), sin solapamiento con el entrenamiento.

### 4.4 Módulo B — Evolución del desgaste con ARIMA([1, 5],1,0)

Implementado en **`notebook_final_predictor_rul.ipynb`**.

- Se modela la serie `log(1 + HI)` del rodamiento 1_2.
- Con una ventana histórica de 30 minutos se ajusta un **ARIMA([1, 5],1,0)** con tendencia lineal y se proyectan los **15 minutos siguientes**, con un intervalo de confianza del 80 %.
- Se evalúa en dos instantes: **t = 41** (FPT, inicio de la degradación) y **t = 115** (degradación avanzada).
- El código incluye un respaldo automático a ARIMA(1,1,0) si el ajuste ARIMA([1, 5],1,0) no converge.

## 5. Resultados

### 5.1 Random Forest — RUL en minutos

| Conjunto | MAE (min) | NMAE (% sobre 60 min) |
|---|---|---|
| **Test** — 30 % final de 1_2, fuera de muestra (37 puntos) | **6,21** | **10,3 %** |
| Train — datos vistos en el entrenamiento | 1,87 | 3,1 % |

Sobre `datos_muestra.csv` (notebook simplificado), el MAE es **6,39 min** en las 30 filas de test y **1,85 min** en las 70 filas de entrenamiento, coherente con los resultados del dataset completo.

**Interpretación:** en la fase final de vida de 1_2, el modelo estima cuántos minutos le quedan al rodamiento con un error medio de unos 6 minutos, lo que permite programar una intervención con margen suficiente antes de la falla.

### 5.2 ARIMA([1, 5],1,0) — Evolución del desgaste (1_2)

| Instante de proyección | Orden usado | MAE de la proyección a 15 min (escala log(HI+1)) |
|---|---|---|
| t = 41 (FPT) | ARIMA([1, 5],1,0) | 0.188 |
| t = 115 (degradación avanzada) | ARIMA([1, 5],1,0) | 0.376 |

La proyección es más precisa en la fase avanzada, donde la tendencia de degradación ya está bien establecida.

### 5.3 Comparación de modelos (validación *leave-one-bearing-out*)

Antes de elegir el modelo final se compararon Ridge y Random Forest dejando **un rodamiento completo fuera** del entrenamiento en cada iteración:

| Rodamiento de test | MAE Ridge (min) | MAE Random Forest (min) |
|---|---|---|
| 1_1 | 32,83 | **15,45** |
| 1_2 | **23,20** | 27,95 |
| 1_3 | 16,45 | **13,43** |

Random Forest fue mejor en 2 de 3 rodamientos y se eligió como modelo final.

## 6. Contenido del ZIP

```
zarate_pablo_estimador_rul_rodamientos.zip
├── README.md                              ← Este archivo
├── notebook_final_simplificado.ipynb      ← Notebook principal de la entrega
├── model.pkl                              ← Modelo Random Forest entrenado
├── datos_muestra.csv                      ← Muestra de 100 filas
├── app.py                                 ← Interfaz web (Streamlit)
├── notebook_final_app_interactiva.ipynb   ← Levanta la interfaz web desde Colab
└── notebook_final_version_completa/
    ├── contenido requerid.bmp             ← Estructura de carpetas que requiere el dataset
    ├── notebook_final_entrenar_modelo.ipynb   ← Pipeline completo de entrenamiento
    └── notebook_final_predictor_rul.ipynb     ← Predicción con el dataset completo + ARIMA
```

| Archivo | Descripción |
|---|---|
| `notebook_final_simplificado.ipynb` | Carga `model.pkl` y `datos_muestra.csv`, predice el RUL, calcula el error por conjunto (train/test), grafica RUL real vs. predicho y muestra tablas comparativas. **No requiere el dataset completo.** |
| `model.pkl` | Diccionario guardado con `joblib` que contiene el modelo (`modelo`), la lista de características (`features`), el tope de RUL (`rul_cap`) y metadatos de la partición train/test y del MAE obtenido. |
| `app.py` + `notebook_final_app_interactiva.ipynb` | Interfaz web donde el usuario elige un caso de `datos_muestra.csv`, ve sus características y obtiene el RUL estimado junto al RUL real, indicando si el caso fue visto o no en el entrenamiento. |
| `notebook_final_entrenar_modelo.ipynb` | Extracción de características, HI, FPT, selección de características, validación leave-one-bearing-out, entrenamiento final y generación de `model.pkl` y `datos_muestra.csv`. |
| `notebook_final_predictor_rul.ipynb` | Recalcula las características desde los CSV crudos, predice el RUL de los tres rodamientos con `model.pkl` (Módulo A) y proyecta la degradación de 1_2 con **ARIMA([1, 5],1,0)** (Módulo B). |

## 7. Cómo ejecutar

Todos los notebooks están preparados para **Google Colab** y usan la ruta `/content/sample_data/`.

### 7.1 Notebook principal (versión simplificada)

1. Abrir `notebook_final_simplificado.ipynb` en Colab.
2. Subir `model.pkl` y `datos_muestra.csv` a la carpeta `/content/sample_data/`.
3. Ejecutar todas las celdas de arriba a abajo (*Entorno de ejecución → Ejecutar todas*).

### 7.2 Interfaz web (Streamlit)

1. Abrir `notebook_final_app_interactiva.ipynb` en Colab.
2. Subir `app.py`, `model.pkl` y `datos_muestra.csv` a `/content/sample_data/`.
3. Ejecutar las celdas en orden. El notebook instala Streamlit y localtunnel, levanta el servicio en el puerto 8501 y muestra:
   - la URL pública (línea `your url is: ...`),
   - la contraseña de acceso del túnel (una dirección IP).
4. Abrir la URL, ingresar la contraseña y probar las predicciones.

> Si la interfaz no carga correctamente o no permite seleccionar un caso, refrescar la página o volver a ejecutar las celdas del túnel. Es una limitación del servicio gratuito localtunnel, suficiente para fines demostrativos.

También puede ejecutarse localmente con `streamlit run app.py` si los tres archivos están en la misma carpeta.

### 7.3 Versión completa (requiere el dataset)

1. Descargar los rodamientos `Bearing1_1`, `Bearing1_2` y `Bearing1_3` del dataset XJTU-SY.
2. Comprimirlos en `Bearing1_1-1_2-1_3.zip` y subirlo a `/content/sample_data/`. El notebook lo descomprime en `/content/sample_data/dataset/`, con la estructura que muestra `contenido requerid.bmp`: una carpeta por rodamiento con los archivos `1.csv`, `2.csv`, ..., `N.csv`. Si las carpetas ya están descomprimidas en esa ruta, el paso se omite.
3. Ejecutar **primero** `notebook_final_entrenar_modelo.ipynb` (genera `model.pkl` y `datos_muestra.csv`, sobrescribiendo los existentes).
4. Ejecutar **después** `notebook_final_predictor_rul.ipynb` (predicción de RUL y proyección ARIMA).

### 7.4 Uso del modelo desde código

```python
import joblib
import pandas as pd

paquete = joblib.load('model.pkl')
modelo, cols = paquete['modelo'], paquete['features']

datos = pd.read_csv('datos_muestra.csv')
datos['rul_predicho'] = modelo.predict(datos[cols])   # RUL estimado en minutos
```

## 8. Dependencias

- Python 3
- `numpy`, `pandas`, `scipy`, `matplotlib`
- `scikit-learn` (PCA, StandardScaler, KFold, Ridge, RandomForestRegressor)
- `statsmodels` (ARIMA)
- `joblib`
- `streamlit` y `localtunnel` (npm), solo para la interfaz web

Todas, salvo Streamlit y localtunnel, vienen preinstaladas en Google Colab.

## 9. Referencias

- Wang, B., Lei, Y., Li, N., & Li, N. (2020). *A Hybrid Prognostics Approach for Estimating Remaining Useful Life of Rolling Element Bearings*. IEEE Transactions on Reliability, 69(1), 401–412.
- Repositorio del dataset XJTU-SY: <https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets>
