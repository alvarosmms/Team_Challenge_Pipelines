import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Transformer que realiza:
# - Escalado de los datos (para clustering)
# - Búsqueda del número óptimo de clusters usando silhouette_score
# - Cálculo de los centros de los clusters y, basándose en los nombres de las columnas, sugiere una etiqueta para cada cluster
# - Transformación de los datos para asignar a cada instancia la etiqueta del cluster al que pertenece
# - Evaluación de los clusters encontrados (silhouette_score, Davies-Bouldin score, tamaños de clusters y características más importantes por cluster)
# - Análisis de perfiles de cluster (tamaño, edad promedio, distribución de gasto, profesiones más comunes, proporción de casados y graduados)
# - Análisis de importancia de características por cluster
    
class ClusteringAndMapping(BaseEstimator, TransformerMixin):
    def __init__(self, k_range=range(2, 11), random_state=42):
        self.k_range = k_range
        self.random_state = random_state

    def fit(self, X, y=None):
        # Si X es DataFrame, se extraen los nombres de las columnas y se obtiene la matriz numérica
        if hasattr(X, 'columns'):
            self.feature_names_ = X.columns.tolist()
            X_numeric = X.values
        else:
            self.feature_names_ = None
            X_numeric = X

        # Escalado de datos
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_numeric)

        best_score = -1
        best_k = None
        best_model = None

        # Se recorre el rango de k para encontrar el que maximice el silhouette_score
        for k in self.k_range:
            model = KMeans(n_clusters=k, random_state=self.random_state)
            labels = model.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_model = model

        self.best_k_ = best_k
        self.best_model_ = best_model
        print("Número óptimo de clusters encontrado:", best_k)

        # Calcular centros de clusters (en espacio escalado)
        self.centers_ = self.best_model_.cluster_centers_
        
        # Ennumeramos los clusters y sugerimos una etiqueta basada en los nombres de las columnas
        self.mapping_ = {}
        for cluster in range(self.best_k_):
            self.mapping_[cluster] = cluster

        print("\n Mapping final de clusters:", self.mapping_)
        return self

    def transform(self, X):
        # Se asegura de aplicar el mismo escalado
        if hasattr(X, 'columns'):
            X_numeric = X.values
        else:
            X_numeric = X
        X_scaled = self.scaler_.transform(X_numeric)
        # Se predicen los clusters usando el modelo óptimo encontrado
        cluster_labels = self.best_model_.predict(X_scaled)
        # Se mapea cada cluster a la etiqueta definida por el usuario o sugerida
        mapped_labels = np.array([self.mapping_[label] for label in cluster_labels])
        return mapped_labels
    
    def evaluate_clusters(X, labels, centers):
        # Silhouette score
        sil_score = silhouette_score(X, labels)
        
        # Davies-Bouldin score
        from sklearn.metrics import davies_bouldin_score
        db_score = davies_bouldin_score(X, labels)
        
        # Calculate cluster sizes
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        
        # Feature importance per cluster
        feature_importance = pd.DataFrame(
            centers,
            columns=X.columns
        ).apply(lambda x: np.abs(x - x.mean()) / x.std())
        
        print(f"Silhouette Score: {sil_score:.3f}")
        print(f"Davies-Bouldin Score: {db_score:.3f}")
        print("\nCluster Sizes:")
        print(cluster_sizes)
        print("\nTop Features per Cluster:")
        for i in range(len(centers)):
            top_features = feature_importance.iloc[i].nlargest(3)
            print(f"\nCluster {i} - Top Features:")
            print(top_features)

    def analyze_cluster_profiles(df, labels):
        df_with_clusters = df.copy()
        df_with_clusters['Cluster'] = labels
        
        profiles = []
        for cluster in np.unique(labels):
            cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
            profile = {
                'Cluster': cluster,
                'Size': len(cluster_data),
                'Avg_Age': cluster_data['Age'].mean(),
                'Spending_Distribution': cluster_data['Spending_Score'].value_counts(normalize=True),
                'Top_Professions': cluster_data['Profession'].value_counts().nlargest(3),
                'Married_Ratio': (cluster_data['Ever_Married'] == 'Yes').mean(),
                'Graduated_Ratio': (cluster_data['Graduated'] == 'Yes').mean()
            }
            profiles.append(profile)
        
        return profiles

    def analyze_feature_importance(pipeline, X):
        # Obtener datos transformados
        preprocessed_data = pipeline.named_steps['preprocessor'].transform(X)
        feature_selected_data = pipeline.named_steps['feature_selector'].transform(preprocessed_data)
        
        # Obtener nombres de características
        numeric_features = ['Age', 'Work_Experience', 'Family_Size', 'Spending_Power']
        categorical_features = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Age_Group']
        
        # Obtener nombres de características después de one-hot encoding
        cat_encoder = pipeline.named_steps['feature_selector'].named_transformers_['cat']
        categorical_features_encoded = cat_encoder.get_feature_names_out(categorical_features)
        
        # Combinar nombres de características
        all_features = numeric_features + categorical_features_encoded.tolist()
        
        # Obtener centroides de clusters
        cluster_centers = pipeline.named_steps['clustering_mapping'].best_model_.cluster_centers_
        
        # Calcular importancia de características por cluster
        for i, center in enumerate(cluster_centers):
            print(f"\nCluster {i} - Características más importantes:")
            # Calcular la distancia desde la media global
            importance = np.abs(center - np.mean(feature_selected_data, axis=0))
            # Obtener top 5 características
            top_features_idx = np.argsort(importance)[-5:]
            for idx in top_features_idx[::-1]:
                print(f"{all_features[idx]}: {importance[idx]:.3f}")

# Transformer que realiza la preprocesamiento de datos:
# - Imputación de valores faltantes
# - Ingeniería de características
# - Conversión de variables categóricas a numéricas

class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.num_features = ['Age', 'Work_Experience', 'Family_Size']
        self.cat_features = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Var_1']
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        
        # Debug print
        print("Columns in input data:", X_copy.columns.tolist())
        
        # Handle missing values
        # Numerical: median imputation
        for col in self.num_features:
            if col in X_copy.columns:
                X_copy[col].fillna(X_copy[col].median(), inplace=True)
        
        # Categorical: mode imputation
        for col in self.cat_features:
            if col in X_copy.columns:
                X_copy[col].fillna(X_copy[col].mode()[0], inplace=True)
        
        # Add feature engineering
        X_copy['Age_Group'] = pd.cut(X_copy['Age'], 
                                    bins=[0, 25, 35, 50, 65, 100],
                                    labels=['Young', 'Young_Adult', 'Adult', 'Senior', 'Elder'])
        
        # Convert Spending_Score to numeric
        spending_map = {'Low': 1, 'Average': 2, 'High': 3}
        X_copy['Spending_Power'] = X_copy['Spending_Score'].map(spending_map)
        
        # Debug print
        print("Columns after preprocessing:", X_copy.columns.tolist())
        
        return X_copy