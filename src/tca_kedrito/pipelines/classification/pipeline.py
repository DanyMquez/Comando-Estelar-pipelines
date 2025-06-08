from kedro.pipeline import Pipeline, node
from .nodes import preprocess_reservation_data, reduce_cardinality, train_svm_model

def create_pipeline(**kwargs):
    return Pipeline([
        node(func=preprocess_reservation_data, inputs="reservas_clustered", outputs="reservas_svm_nocard", name="preprocess_for_svm"),
        node(func=reduce_cardinality, inputs="reservas_svm_nocard", outputs="reservas__svm_card", name="reduce_cardinality_for_svm"),
        node(func=train_svm_model, inputs="reservas__svm_card", outputs="svm_model", name="train_svm_model")
        ])
