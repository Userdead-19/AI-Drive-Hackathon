from google.cloud import storage


def download_dataset(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(
        f"Downloaded storage object {source_blob_name} from bucket {bucket_name} to local file {destination_file_name}."
    )


def list_blobs(bucket_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs()
    return [blob.name for blob in blobs]
