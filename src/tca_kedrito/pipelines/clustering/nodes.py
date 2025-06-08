from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
import pandas as pd
import mlflow


def apply_gmm(df: pd.DataFrame, X_pca) -> pd.DataFrame:
    df = df.copy()
    import mlflow.sklearn

    # Definir el grid de parámetros
    param_grid = {
        "n_components": [4], # Ajustar este rango según necesidades
        "covariance_type": ["spherical", "tied", "diag", "full"], # Tipos de covarianza a probar
    }
    best_score = -1
    best_params = None

    # Buscar los mejores parámetros usando silhouette score
    for n in param_grid["n_components"]:
        for cov in param_grid["covariance_type"]:
            gmm = GaussianMixture(n_components=n, covariance_type=cov)
            labels = gmm.fit_predict(X_pca)
            score = silhouette_score(X_pca, labels)
            if score > best_score:
                best_score = score
                best_params = {"n_components": n, "covariance_type": cov}

    # Ajustar el modelo GMM con los mejores parámetros
    n_clusters_gmm = best_params['n_components']
    covariance_type = best_params['covariance_type']
    gmm = GaussianMixture(n_components=n_clusters_gmm, covariance_type=covariance_type, random_state=42)
    clusters_gmm = gmm.fit_predict(X_pca)
    df['cluster_gmm'] = clusters_gmm

    silhouette_gmm = silhouette_score(X_pca, clusters_gmm)

    # Registrar el modelo en MLflow
    with mlflow.start_run(run_name="GMM Clustering"):
        mlflow.log_params({
            "n_components": n_clusters_gmm,
            "covariance_type": covariance_type
        })
        mlflow.log_metric("silhouette_score", silhouette_gmm)
        mlflow.sklearn.log_model(gmm, "gmm_model")

    return df

