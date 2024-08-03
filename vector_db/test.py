# import os
# from google.cloud import storage

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
#     "D:\\Exploration\\AI-Drive-Hackathon\\vector_db\\ai-drive-psg-2024-local-sa.json"
# )


# def list_blobs_with_prefix(bucket_name, prefix, delimiter=None):
#     """Lists all the blobs in the bucket that begin with the prefix."""
#     storage_client = storage.Client()

#     blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)

#     print("Blobs:")
#     for blob in blobs:
#         print(blob.name)

#     if delimiter:
#         print("Prefixes:")
#         for prefix in blobs.prefixes:
#             print(prefix)


# bucket_name = "ai-drive-psg-2024-us-central1-public"
# prefix = "test_scenario/"

# list_blobs_with_prefix(bucket_name, prefix)

# import pandas as pd
# import numpy as np


# # Load the feather file
# file_path = (
#     "D:\\Exploration\\AI-Drive-Hackathon\\vector_db\\tmp\\local_dataset_1.feather"
# )
# df = pd.read_feather(file_path)

# # Get the dimensionality of the first embedding
# first_embedding = df["embeddings"].iloc[0]
# embedding_dim = (
#     first_embedding.shape[0] if isinstance(first_embedding, np.ndarray) else None
# )
# print(f"Dimensionality of embeddings: {embedding_dim}")


arr = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    0.11,
    0.22,
    0.33,
    0.44,
    0.55,
    0.66,
    0.77,
    0.88,
    0.99,
    1.1,
    0.12,
    0.23,
    0.34,
    0.45,
    0.56,
    0.67,
    0.78,
    0.89,
    1.0,
    1.1,
    0.13,
    0.24,
    0.35,
    0.46,
    0.57,
    0.68,
    0.79,
    0.80,
    0.91,
    1.2,
    0.14,
    0.25,
    0.36,
    0.47,
    0.58,
    0.69,
    0.70,
    0.81,
    0.92,
    1.3,
    0.15,
    0.26,
    0.37,
    0.48,
    0.59,
    0.60,
    0.71,
    0.82,
    0.93,
    1.4,
    0.16,
    0.27,
    0.38,
    0.49,
    0.50,
    0.61,
    0.72,
    0.83,
    0.94,
    1.5,
    0.17,
    0.28,
    0.39,
    0.40,
    0.51,
    0.62,
    0.73,
    0.84,
    0.95,
    1.6,
    0.18,
    0.29,
    0.30,
    0.41,
    0.52,
    0.63,
    0.74,
    0.85,
    0.96,
    1.7,
    0.19,
    0.30,
    0.31,
    0.42,
    0.53,
    0.64,
    0.75,
    0.86,
    0.97,
    1.8,
    0.20,
    0.31,
    0.32,
    0.43,
    0.54,
    0.65,
    0.76,
    0.87,
    0.98,
    1.9,
    0.86,
    0.97,
    1.8,
    0.20,
    0.31,
    0.32,
    0.43,
    0.54,
    0.65,
    0.76,
    0.87,
    0.98,
    0.86,
    0.97,
    1.8,
    0.20,
    0.31,
    0.32,
]

print(len(arr))
