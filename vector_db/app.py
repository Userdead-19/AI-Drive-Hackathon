import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from google.cloud import storage
import logging
import threading


app = Flask(__name__)

data_cache = None
cache_lock = threading.Lock()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    "D:\\Exploration\\AI-Drive-Hackathon\\vector_db\\ai-drive-psg-2024-local-sa.json"
)


def download_dataset(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    logger.info(
        f"Downloaded storage object {source_blob_name} from bucket {bucket_name} to local file {destination_file_name}."
    )


def load_dataset(file_path):
    global data_cache
    with cache_lock:
        if data_cache is None:
            df = pd.read_feather(file_path)
            data_cache = np.vstack(df["embeddings"].values)
        return data_cache


def dot_product_knn(query_vector, data, k=10):
    similarities = np.dot(data, query_vector)
    indices = np.argsort(-similarities)[:k]  # Get indices of top k largest values
    return indices.tolist()


@app.route("/KNN_search", methods=["POST"])
def knn_search():
    request_data = request.get_json()
    dataset_id = request_data["dataset_id"]
    query_vector = np.array(request_data["query_vector"])

    # Assuming dataset is stored in GCS bucket named 'ai-drive-psg-2024-us-central1-public'
    bucket_name = "ai-drive-psg-2024-us-central1-public"
    source_blob_name = f"{dataset_id}"
    destination_file_name = f"tmp/{dataset_id}"
    print(destination_file_name)
    os.makedirs("tmp", exist_ok=True)  # Ensure the directory exists
    download_dataset(bucket_name, source_blob_name, destination_file_name)
    data = load_dataset(destination_file_name)

    # Ensure query_vector has the same dimensionality as the embeddings
    if query_vector.shape[0] != data.shape[1]:
        return (
            jsonify(
                {
                    "error": "Query vector dimensionality does not match dataset embeddings."
                }
            ),
            400,
        )

    indices = dot_product_knn(query_vector, data, k=10)

    return jsonify({"nearest_neighbor": {"indices": indices}})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
