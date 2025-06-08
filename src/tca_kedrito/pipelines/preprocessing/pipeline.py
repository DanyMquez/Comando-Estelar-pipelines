from kedro.pipeline import Pipeline, node
from .nodes import load_and_clean_data, remove_duplicated_columns, cardinality, exhaustive_preprocess, prep_for_model, scale_and_encode, find_pca_elbow, apply_pca

def create_pipeline(**kwargs):
    return Pipeline([
        node(load_and_clean_data, inputs="iar_Reservaciones", outputs="reservas_clean", name="clean_data"),
        node(remove_duplicated_columns, inputs="reservas_clean", outputs="reservas_clean_dedup", name="dedup_columns"),
        node(cardinality, inputs="reservas_clean_dedup", outputs="reservas_nocard", name="cardinality_check"),
        node(exhaustive_preprocess, inputs="reservas_nocard", outputs="reservas_engineered", name="feature_engineering"),
        node(prep_for_model, inputs="reservas_engineered", outputs="reservas_prepped", name="preparation_for_model"),
        node(scale_and_encode, inputs="reservas_prepped", outputs="reservas_scaled_encoded", name="scale_and_encode"),
        node(find_pca_elbow, inputs="reservas_scaled_encoded", outputs="pca_elbow_value", name="find_pca_elbow"),
        node(apply_pca, inputs=["reservas_scaled_encoded","pca_elbow_value"], outputs="reservas_pca", name="apply_pca")

    ])