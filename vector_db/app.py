import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from google.cloud import storage
import logging
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os

from flask_caching import Cache

app = Flask(__name__)
cache = Cache(config={"CACHE_TYPE": "simple"})
cache.init_app(app)

executor = ThreadPoolExecutor(max_workers=10)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    "D:\\Exploration\\AI-Drive-Hackathon\\vector_db\\ai-drive-psg-2024-local-sa.json"
)


data_cache = None
cache_lock = threading.Lock()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Asynchronous download
async def download_dataset(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, blob.download_to_filename, destination_file_name)
    logger.info(
        f"Downloaded storage object {source_blob_name} from bucket {bucket_name} to local file {destination_file_name}."
    )
    if os.path.exists(destination_file_name):
        file_size = os.path.getsize(destination_file_name)
        logger.info(f"Downloaded file size: {file_size} bytes")


def validate_feather_file(file_path):
    try:
        with open(file_path, "rb") as f:
            if f.read(8) != b"FEA1".ljust(8, b"\0"):
                return False
        return True
    except Exception as e:
        logger.error(f"File validation error: {e}")
        return False


def load_dataset(file_path):
    global data_cache
    with cache_lock:
        if data_cache is None:
            if not validate_feather_file(file_path):
                raise ValueError(f"The file {file_path} is not a valid Feather file.")
            df = pd.read_feather(file_path)
            data_cache = np.vstack(df["embeddings"].values)
        return data_cache


def dot_product_knn(query_vector, data, k=10):
    similarities = np.dot(data, query_vector)
    indices = np.argsort(-similarities)[:k]
    return indices.tolist()


@app.route("/KNN_search", methods=["POST"])
@cache.cached(timeout=60, key_prefix="knn_search")
async def knn_search():
    try:
        request_data = request.get_json()
        dataset_id = request_data["dataset_id"]
        query_vector = np.array(request_data["query_vector"])

        bucket_name = "your-bucket-name"
        source_blob_name = f"{dataset_id}.feather"
        destination_file_name = f"tmp/{dataset_id}.feather"
        os.makedirs("tmp", exist_ok=True)

        if (
            not os.path.exists(destination_file_name)
            or os.path.getsize(destination_file_name) == 0
        ):
            await download_dataset(bucket_name, source_blob_name, destination_file_name)

        data = load_dataset(destination_file_name)

        if query_vector.shape[0] != data.shape[1]:
            return (
                jsonify(
                    {
                        "error": "Query vector dimensionality does not match dataset embeddings."
                    }
                ),
                400,
            )

        loop = asyncio.get_event_loop()
        indices = await loop.run_in_executor(
            executor, dot_product_knn, query_vector, data, 10
        )

        return jsonify({"nearest_neighbor": {"indices": indices}})
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
