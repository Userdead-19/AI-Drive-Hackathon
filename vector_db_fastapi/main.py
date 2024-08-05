import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from google.cloud import storage
import logging
import os
import asyncio
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from redis import Redis
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=20)

data_cache = None
cache_lock = asyncio.Lock()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    "D:\\Exploration\\AI-Drive-Hackathon\\vector_db\\ai-drive-psg-2024-local-sa.json"
)

redis = Redis(host="localhost", port=6379, db=0)


async def download_dataset(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    blob = storage_client.bucket(bucket_name).blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    logger.info(f"Downloaded file to {destination_file_name}")


def validate_feather_file(file_path):
    try:
        with open(file_path, "rb") as f:
            if f.read(8) != b"FEA1".ljust(8, b"\0"):
                return False
        return True
    except Exception as e:
        logger.error(f"File validation error: {e}")
        return False


async def load_dataset(file_path):
    global data_cache
    async with cache_lock:
        if data_cache is None:
            # if not validate_feather_file(file_path):
            #     raise ValueError(f"The file {file_path} is not a valid Feather file.")
            df = pd.read_feather(file_path)
            data_cache = np.vstack(df["embeddings"].values)
        return data_cache


def dot_product_knn(query_vector, data, k=10):
    similarities = np.dot(data, query_vector)
    indices = np.argsort(-similarities)[:k]
    return indices.tolist()


@app.on_event("startup")
async def on_startup():
    FastAPICache.init(RedisBackend(redis))
    logger.info("Cache backend initialized")


@app.post("/KNN_search")
@cache(expire=60)  # Caches the result of this endpoint
async def knn_search(request: Request):
    try:
        request_data = await request.json()
        dataset_id = request_data["dataset_id"]
        query_vector = np.array(request_data["query_vector"])

        bucket_name = "your-bucket-name"
        source_blob_name = f"{dataset_id}"
        destination_file_name = f"tmp/{dataset_id}"
        os.makedirs("tmp", exist_ok=True)

        if (
            not os.path.exists(destination_file_name)
            or os.path.getsize(destination_file_name) == 0
        ):
            await download_dataset(bucket_name, source_blob_name, destination_file_name)

        data = await load_dataset(destination_file_name)

        if query_vector.shape[0] != data.shape[1]:
            return {
                "error": "Query vector dimensionality does not match dataset embeddings."
            }, 400

        loop = asyncio.get_event_loop()
        indices = await loop.run_in_executor(
            executor, dot_product_knn, query_vector, data, 10
        )

        return {"nearest_neighbor": {"indices": indices}}
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return {"error": str(e)}, 500


@app.get("/")
def read_root():
    return {"Hello": "World"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000, workers=4)
