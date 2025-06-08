from kedro.pipeline import Pipeline, node
from .nodes import apply_gmm

def create_pipeline(**kwargs):
    return Pipeline([
        node(apply_gmm, inputs=["reservas_engineered","reservas_pca"], outputs="reservas_clustered", name="apply_gmm")
    ])