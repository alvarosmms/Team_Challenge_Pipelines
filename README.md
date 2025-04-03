

<h1 align="center">🔄 ML Pipelines - Team Challenge 🔄</h1>

## <div align="center"> 🤖 Modelos Supervisados y No Supervisados 🤖</div>

### <div align="center">📊 Descripción del Proyecto 📊</div>

Este Team Challenge se centra en la práctica y construcción de **Pipelines de Scikit-learn** para procesar datos, entrenar modelos y evaluar su rendimiento de manera eficiente y reproducible. 

El objetivo es encapsular los pasos de preprocesamiento y modelado dentro de pipelines, aprovechando sus ventajas en **validación cruzada**, **prevención de data leakage** y **optimización de hiperparámetros**.

-------------------------

### 🎯 **Objetivos del Proyecto** 🎯

1. Implementar **Pipelines completos** para modelos **supervisados y no supervisados**.
2. Aplicar **OneHotEncoder** para manejar variables categóricas sin errores entre conjuntos de entrenamiento y prueba.
3. Utilizar **validación cruzada** para evaluar modelos y demostrar sus ventajas al usar pipelines.
4. Optimizar hiperparámetros mediante **GridSearchCV**.
5. Opcional: Implementar el proyecto en **entornos virtuales** con archivos `requirements.txt` y `.gitignore`.

-------------------------

### 📊 **Metodología** 📊

**1️⃣ Clustering (No Supervisado)**
- Identificación del número óptimo de clusters mediante métricas de validación.
- Asignación de etiquetas basadas en centroides.

**2️⃣ Preprocesamiento de Datos (Supervisado)**
- Uso de `ColumnTransformer` y `OneHotEncoder` para transformar datos categóricos y numéricos.
- Normalización y escalado de variables numéricas.

**3️⃣ Entrenamiento y Evaluación de Modelos**
- Implementación de **pipelines completos** con `GridSearchCV`.
- Comparación de modelos mediante validación cruzada.

-------------------------

### 📌 Contexto y Relevancia 📌

El uso de **pipelines en Machine Learning** permite construir modelos eficientes y reutilizables, asegurando que cada paso del preprocesamiento se aplique de manera consistente. Esto es clave en entornos de producción y facilita la automatización del flujo de datos.

-------------------------

### 📂 **Estructura del Repositorio** 📂

```
├── README.md
├── requirements.txt
├── .gitignore
└── src
    ├── data
    │   ├── Train.csv
    │   └── Test.csv
    ├── models
    │   └── (modelos entrenados se guardarán aquí)
    ├── result_notebooks
    │   ├── Plumbers_Enterprise_Pipelines_I.ipynb  # Entrenamiento y guardado del modelo
    │   └── Plumbers_Enterprise_Pipelines_II.ipynb # Carga del modelo y predicciones
    ├── notebooks                                  # Pruebas y exploración
    └── utils                                      # Funciones auxiliares
```

-------------------------

### 📌 **Instrucciones de Uso** 📌

#### 🔹 **Creación del Entorno Virtual**

```bash
python3 -m venv venv
source venv/bin/activate   # En Linux/macOS
venv\Scripts\activate      # En Windows
```

#### 🔹 **Instalación de Dependencias**

```bash
pip install -r requirements.txt
```

#### 🔹 **Ejecución del Proyecto**

1. **Entrenamiento del Modelo:** Ejecuta [`Plumblers_Enterprise_Pipelines_I.ipynb`](src/result_notebooks/<nombre_grupo>_Pipelines_I.ipynb) para entrenar el modelo y guardarlo en `/src/models`.
2. **Predicción y Evaluación:** Ejecuta [`Plumblers_Enterpise_Pipelines_II.ipynb`](src/result_notebooks/<nombre_grupo>_Pipelines_II.ipynb) para realizar predicciones y evaluar los resultados.

-------------------------

### 📈 **Conclusiones y Recomendaciones** 📈

1. **Eficiencia en Pipelines:** La encapsulación de los pasos en un solo flujo mejora la reproducibilidad y evita *data leakage*.
2. **Comparación de Modelos:** Modelos como **XGBoost y RandomForest** han mostrado un mejor desempeño en términos de precisión y generalización.
3. **Clusterización Interactiva:** La asignación de etiquetas basada en centroides permite mejorar la interpretabilidad del modelo no supervisado.
4. **Automatización y Escalabilidad:** La implementación de estos pipelines facilita su integración en entornos de producción.

-------------------------

### 🔧 Construido con 🔧

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/Numpy-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-003b57?style=flat-square&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-AA2222?style=flat-square&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/en/stable/)

-------------------------

### ✍️ Autoría ✍️

👥 **Nombre del equipo:** Plumbers_Enterprise  
🧑‍💻 **Integrantes:**  
- Álvaro Sánchez  
- Juan Moreno  
- Xián Mosquera  
- Vicky Sequeira  
- Dani Castillo  

-------------------------

¡Gracias por revisar nuestro proyecto! 
