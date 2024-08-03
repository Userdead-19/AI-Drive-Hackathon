from google.cloud import storage


def list_datasets(bucket_name):
    """Lists all the blobs in the bucket."""
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name)

    datasets = [blob.name for blob in blobs]
    return datasets


if __name__ == "__main__":
    bucket_name = "ai-drive-psg-2024-us-central1"
    datasets = list_datasets(bucket_name)

    for dataset in datasets:
        print(dataset)
