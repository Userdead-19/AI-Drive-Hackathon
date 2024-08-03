import numpy as np


def dot_product_knn(query_vector, data, k):
    similarities = np.dot(data, query_vector)
    nearest_indices = np.argsort(-similarities)[:k]
    return nearest_indices.tolist()


def load_dataset(file_path):
    # Assuming data is stored in Feather format
    import pandas as pd

    data = pd.read_feather(file_path)
    return data.to_numpy()
