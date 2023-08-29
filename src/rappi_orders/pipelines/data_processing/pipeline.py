from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_orders, calculate_features_stores, create_model_input_table


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_orders,
                inputs="orders",
                outputs="preprocessed_orders",
                name="preprocess_orders_node",
            ),
            node(
                func=calculate_features_stores,
                inputs="preprocessed_orders",
                outputs="store_features",
                name="calculate_features_stores_node",
            ),
            node(
                func=create_model_input_table,
                inputs=["preprocessed_orders", "store_features"],
                outputs="model_input_table",
                name="create_model_input_table_node",
            ),
        ]
    )
