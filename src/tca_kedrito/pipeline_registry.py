"""Project pipelines."""
from kedro.pipeline import Pipeline
from tca_kedrito.pipelines import preprocessing, clustering, classification

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "preprocessing": preprocessing.create_pipeline(),
        "clustering": clustering.create_pipeline(),
        "classification": classification.create_pipeline(),
        "__default__": (
            preprocessing.create_pipeline()
            + clustering.create_pipeline()
            + classification.create_pipeline()
        )
    }
