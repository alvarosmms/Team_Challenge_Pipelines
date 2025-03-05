# Team_Challenge_Pipelines

**Nombre del grupo:** Plumbers_Enterprise 
**Integrantes:**  
- Álvaro Sánchez  
- Juan Moreno  
- Xián Mosquera  
- Vicky Sequeira
- Dani Castillo

---

## Objetivo del Proyecto

Este Team Challenge tiene como objetivo desarrollar y comparar pipelines completos para la construcción de modelos de Machine Learning. Se ha implementado un pipeline para:

- **Procesamiento no supervisado:**  
  Identificación de clusters en datos sin etiquetar, búsqueda del número óptimo de clusters mediante *silhouette score* y mapeo interactivo de clusters a etiquetas (con sugerencia basada en los datos).

- **Procesamiento supervisado:**  
  Preprocesado (usando `ColumnTransformer` y `OneHotEncoder`), optimización de hiperparámetros y evaluación de varios modelos (incluyendo LogisticRegression, RandomForest, XGBoost, LightGBM y CatBoost) mediante validación cruzada y GridSearchCV.

El objetivo es demostrar cómo encapsular todos los pasos relevantes dentro de pipelines para evitar *data leakage* y asegurar la reproducibilidad del modelo.

---

## Estructura del Repositorio

    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    └── src
        ├── data
        │   ├── <nombre_dataset>_train.csv
        │   └── <nombre_dataset>_test.csv
        ├── models
        │   └── (modelos entrenados se guardarán aquí)
        ├── result_notebooks
        │   ├── <nombre_grupo>_Pipelines_I.ipynb      # Notebook de entrenamiento y guardado del modelo
        │   └── <nombre_grupo>_Pipelines_II.ipynb     # Notebook para cargar el modelo y realizar predicciones
        ├── notebooks         (opcional: notebooks de pruebas y exploración)
        └── utils             (opcional: librerías o funciones auxiliares)
---

## Entorno Virtual y Dependencias

El proyecto ha sido desarrollado utilizando **Python 3.12.6** (ajústese a la versión que se esté utilizando). Todas las dependencias necesarias se encuentran listadas en el archivo [`requirements.txt`](requirements.txt).

### Creación y Activación del Entorno Virtual (usando `venv`)

      bash
      python3 -m venv venv
      source venv/bin/activate   # En Linux/macOS
      venv\Scripts\activate      # En Windows

### Instalación de Dependencias

      bash
      pip install -r requirements.txt

---

## Ejecución del Proyecto

1. **Entrenamiento y Guardado del Modelo:**  
   Abre y ejecuta el notebook [`<nombre_grupo>_Pipelines_I.ipynb`](src/result_notebooks/<nombre_grupo>_Pipelines_I.ipynb) para entrenar el pipeline completo. Al finalizar, el modelo entrenado se guardará en el directorio `/src/models`.

2. **Predicción y Evaluación:**  
   Abre y ejecuta el notebook [`<nombre_grupo>_Pipelines_II.ipynb`](src/result_notebooks/<nombre_grupo>_Pipelines_II.ipynb) para cargar el modelo guardado, realizar predicciones sobre `<nombre_dataset>_test.csv` y evaluar los resultados mediante métricas y visualizaciones (matriz de confusión, curva ROC, etc).

---

## Instrucciones Adicionales

- **Visualizaciones Condicionales:**  
  Las visualizaciones (por ejemplo, la curva ROC) se adaptan automáticamente según se trate de un problema binario o multiclase.

- **Mapeo Interactivo de Clusters:**  
  Durante la ejecución del pipeline de clustering se solicitará al usuario la asignación de etiquetas a cada cluster. Se sugerirá una etiqueta basada en la información del centro del cluster y los nombres de las columnas, que podrá ser aceptada o modificada por el usuario.

- **Modelos Supervisados:**  
  El pipeline de la fase supervisada incluye la optimización de hiperparámetros mediante GridSearchCV, y se comparan modelos de LogisticRegression, RandomForest, XGBoost, LightGBM y CatBoost.

---

## Notas

- Asegúrate de tener instaladas todas las dependencias especificadas en `requirements.txt`.
- El archivo `.gitignore` se encuentra configurado para excluir archivos y directorios generados automáticamente (por ejemplo, entornos virtuales, archivos temporales, etc).

---

¡Gracias por revisar nuestro proyecto!
