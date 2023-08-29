import pandas as pd

def preprocess_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for orders.

    Args:
        orders: Raw data.
    Returns:
        Preprocessed data without duplicates and converted `created_at` to datetime.
    """

    # Eliminar registros identicos
    orders = orders.drop_duplicates()

    # Eliminar fechas invalidas
    orders = orders[orders['created_at'].str.len() == 20]

    # Casteo la fecha y extraigo la hora
    orders.loc[:, 'created_at_dt'] = pd.to_datetime(orders['created_at'])
    orders['hour'] = orders['created_at_dt'].dt.hour

    return orders

def preprocess_stores(orders: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for stores.

    Args:
        orders: Raw data.
    Returns:
        Preprocessed data.
    """

    # Eliminar registros identicos
    stores = orders.copy()
    stores = stores.drop_duplicates()

    # Eliminar fechas invalidas
    stores = stores[stores['created_at'].str.len() == 20]
    
    return stores

def calculate_features_stores(preprocessed_stores: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for stores.

    Args:
        orders: Raw data.
    Returns:
        Preprocessed data, with new features.
    """

    # Calcular features
    stores = pd.DataFrame(preprocessed_stores.groupby('store_id')['taken'].agg(taken_percentage='mean') * 100).reset_index()

    return stores

def create_model_input_table(orders: pd.DataFrame, store_features: pd.DataFrame) -> pd.DataFrame:
    """Combines all data to create a model input table.

    Args:
        orders: Preprocessed data for orders.
        store_features: Features for stores.
    Returns:
        Model input table.

    """
    model_input_table = orders.merge(store_features, left_on="store_id", right_on="store_id")
    model_input_table = model_input_table.dropna()
    return model_input_table
