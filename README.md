# Taller B4-T1 — Diseño de Redes Confiables

Proyecto académico de clasificación de riesgo de impago sobre el dataset **Home Credit**. El trabajo desarrolla y compara redes neuronales con arquitectura personalizada, aprendizaje justo, optimización automática e incertidumbre predictiva.

La implementación completa se encuentra en [`T1_B4.ipynb`](T1_B4.ipynb).

## Objetivos

- Analizar y preparar los datos de solicitudes de crédito.
- Construir un modelo base de clasificación binaria.
- Diseñar una capa Keras personalizada con ratios financieros interpretables.
- Incorporar una **FAIR Loss** que penalice la dependencia entre la predicción y el género.
- Buscar hiperparámetros con **Keras Tuner** y estudiar el compromiso entre precisión y justicia.
- Estimar la incertidumbre mediante **Monte Carlo Dropout**.

## Metodología

### Preparación de los datos

El notebook utiliza nueve columnas de `application_train.csv`:

- `TARGET`: variable objetivo (`1` indica dificultades de pago).
- `CODE_GENDER`: variable sensible.
- `AMT_INCOME_TOTAL`, `AMT_CREDIT` y `AMT_ANNUITY`: variables financieras.
- `DAYS_BIRTH`.
- `EXT_SOURCE_1`, `EXT_SOURCE_2` y `EXT_SOURCE_3`.

El preprocesado incluye imputación por mediana, conversión de edad, codificación del género, escalado con `StandardScaler` y una división reproducible **80 % train / 10 % validación / 10 % test**.

Debido al desbalanceo de `TARGET`, el entrenamiento utiliza pesos de clase calculados con `compute_class_weight`.

### Arquitectura personalizada

`RatioFinancieroLayer` calcula dos ratios sobre las variables financieras sin escalar:

- **DSR**: anualidad / ingresos.
- **LTI**: crédito / ingresos.

Los ratios pasan por una saturación sigmoide con pendiente y umbral entrenables. Sus parámetros están limitados mediante el constraint personalizado `CustomClipValue`.

### Aprendizaje justo

La pérdida personalizada combina clasificación y justicia:

```text
FAIR Loss = BCE ponderada + lambda_fair × correlación(predicción, género)²
```

El género se empaqueta junto a la etiqueta para que la función de coste pueda calcular la penalización. La evaluación FAIR utiliza:

- Correlación absoluta entre probabilidad predicha y género.
- Brecha de paridad demográfica entre los dos grupos sensibles.

### AutoML

Keras Tuner explora distintas topologías, dropout y valores de `lambda_fair`. Los modelos se comparan mediante una curva de Pareto entre rendimiento predictivo y dependencia FAIR.

### Incertidumbre

El bloque de Monte Carlo Dropout realiza múltiples predicciones con dropout activo. La media representa la probabilidad final y la varianza se utiliza como medida de incertidumbre.

## Estructura del proyecto

```text
taller-aml-miax/
├── data/
│   └── application_train.csv       # Dataset gestionado con Git LFS
├── reports/
│   ├── figuras/                    # Gráficos generados por el notebook
│   └── *.csv                       # Tablas y resultados
├── T1_B4.ipynb                     # Notebook principal
├── Lectura_datos_Taller_B4_T1.ipynb
├── Taller_B4_T1.pdf                # Enunciado de la práctica
├── requirements.txt
├── .gitattributes                  # Configuración de Git LFS
└── README.md
```

La carpeta `kt_tuner/` se genera localmente durante la búsqueda de hiperparámetros y no se versiona.

## Requisitos

- Python **3.11** (probado con Python 3.11.9).
- Git LFS.
- Memoria suficiente para trabajar con un CSV de aproximadamente 166 MB.

Las dependencias principales son:

- NumPy y pandas.
- scikit-learn.
- Matplotlib y Seaborn.
- TensorFlow y Keras.
- Keras Tuner.
- Jupyter y TensorBoard.

## Instalación

Clonar el repositorio y descargar el dataset gestionado con Git LFS:

```powershell
git clone https://github.com/ivan-nevado/taller-aml-miax.git
cd taller-aml-miax
git lfs install
git lfs pull
```

Crear y activar el entorno virtual en Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

En Linux o macOS, la activación se realiza con:

```bash
source .venv/bin/activate
```

## Ejecución

Iniciar Jupyter:

```powershell
jupyter notebook T1_B4.ipynb
```

También puede utilizarse JupyterLab:

```powershell
jupyter lab
```

Las celdas deben ejecutarse **en orden**, ya que los bloques posteriores reutilizan los datos, modelos y funciones definidos anteriormente.

El notebook busca el dataset principalmente en:

```text
data/application_train.csv
```

## Resultados actuales

Los siguientes valores proceden de los CSV generados en `reports/`. Pueden variar ligeramente al repetir el entrenamiento.

### Modelo base frente a arquitectura custom

| Modelo | Accuracy | ROC-AUC | F1 |
|---|---:|---:|---:|
| Base | 0.6963 | 0.7404 | 0.2568 |
| Custom | 0.6941 | 0.7383 | 0.2550 |

La capa personalizada mantiene un rendimiento cercano al modelo base y aporta ratios financieros interpretables.

### Efecto de la FAIR Loss

| Modelo | Accuracy | ROC-AUC | F1 | \|Corr(pred, género)\| | Brecha de paridad |
|---|---:|---:|---:|---:|---:|
| Custom con BCE | 0.6941 | 0.7383 | 0.2550 | 0.2227 | 0.0891 |
| Custom + FAIR (`lambda=0.2`) | 0.6971 | 0.7352 | 0.2536 | 0.0630 | 0.0248 |

La FAIR Loss reduce claramente la dependencia respecto del género, con una disminución pequeña del ROC-AUC.

### Mejor compromiso obtenido con Keras Tuner

| Modelo | Accuracy | ROC-AUC | F1 | \|Corr(pred, género)\| | Brecha de paridad |
|---|---:|---:|---:|---:|---:|
| Custom + FAIR Tuner (`lambda≈0.5861`) | 0.6929 | 0.7338 | 0.2530 | 0.0207 | 0.0080 |

Este resultado prioriza una dependencia FAIR muy baja a cambio de una reducción moderada del rendimiento predictivo.

## Salidas generadas

Al ejecutar el notebook se crean, entre otras:

- Curvas de entrenamiento y validación.
- Matrices de confusión.
- Gráficos del análisis exploratorio.
- Curva de Pareto de precisión frente a dependencia FAIR.
- Distribuciones de incertidumbre de MC Dropout.
- Tablas CSV con métricas y resultados del tuner.

## Reproducibilidad

- Semilla global: `42`.
- Split fijo: 80/10/10.
- `StandardScaler` ajustado únicamente con train.
- Early stopping con restauración de los mejores pesos.
- El resultado puede presentar pequeñas diferencias numéricas según la versión de TensorFlow, el sistema operativo y el hardware.

