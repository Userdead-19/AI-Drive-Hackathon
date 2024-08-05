import requests
import time
from google.cloud import storage
import os
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed


def check_result():

    url = "http://localhost:5000/KNN_search"
    query_vector = [
        0.5192618794218221,
        0.45182743160924277,
        0.6186627853125054,
        0.6589039736527896,
        0.2437497527606538,
        0.6322012954709343,
        0.6957154561999354,
        0.3211021082103015,
        0.5171008074178978,
        0.4417884791762957,
        0.6613229795135122,
        0.48464523291490813,
        0.6497945195170712,
        0.23912660006608044,
        0.4630773135161953,
        0.59790179792904,
        0.8568005146522668,
        0.016867737472978095,
        0.33021050861861156,
        0.7381802230880444,
        0.330048888486159,
        0.04099779082555588,
        0.8830641035390772,
        0.9076208617180413,
        0.3684242511943032,
        0.603300031335987,
        0.22937810100759792,
        0.38553808649945565,
        0.3263111342891757,
        0.648878150164451,
        0.27033621879561787,
        0.5984316133463654,
        0.2786551353575998,
        0.708239720349325,
        0.19557054351197933,
        0.2124901054229329,
        0.961868425730306,
        0.08016004450901215,
        0.8818673981954538,
        0.7896991041465541,
        0.9372265692560704,
        0.7363241368240393,
        0.8489754495572857,
        0.19005732400141429,
        0.7867922270416392,
        0.8427674967911334,
        0.6932224997029082,
        0.20526093268371848,
        0.6625187328939475,
        0.14529154933731114,
        0.017453104759061033,
        0.841529155394089,
        0.5019500533911736,
        0.4437103475463062,
        0.9072997018291746,
        0.5744854760228293,
        0.9879924406985708,
        0.11493472031126228,
        0.3207655848576144,
        0.4626453996499891,
        0.2149679242659246,
        0.01077809465642987,
        0.1807329316074454,
        0.9253217896438373,
        0.1233506329869617,
        0.6047319598307184,
        0.7627409268198013,
        0.5658423952274366,
        0.7848895385977217,
        0.6577608653256217,
        0.2733850998191366,
        0.17567115102587583,
        0.5636211361968779,
        0.5924207324966091,
        0.5563059375161599,
        0.37678961656349663,
        0.9128033951935293,
        0.739434851543232,
        0.7059990842746358,
        0.22319770639152192,
        0.88615302811347,
        0.6950836864771088,
        0.24032852124458803,
        0.2281443005968229,
        0.983969324885417,
        0.4002245404880236,
        0.02057674934900544,
        0.7405818122684916,
        0.06393126348996403,
        0.5033455362160272,
        0.704874775375181,
        0.7376453247978968,
        0.4890926218506715,
        0.8383124429343343,
        0.5096920698942622,
        0.21265594441822655,
        0.4852562435927922,
        0.39147699126701185,
        0.5743457175533937,
        0.3888956520131691,
        0.5737552070504911,
        0.5572852886563897,
        0.38168985736534733,
        0.2796519395502186,
        0.9002948238707822,
        0.9153197020558731,
        0.2675276355542606,
        0.43516799482770463,
        0.6709871806316015,
        0.09189965591387383,
        0.18775322526771687,
        0.49757739710012916,
        0.2645309367497458,
        0.11707052844911137,
        0.7683888666451091,
        0.5806566344326539,
        0.32407981345403725,
        0.8894339622784585,
        0.3415895981987198,
        0.07335606390491922,
        0.4021511845068496,
        0.8902462604197977,
        0.893165186733412,
        0.3021460567445712,
        0.9588496764062373,
        0.9296382917067266,
        0.6706809297323028,
        0.4219199327363604,
    ]
    blob_name = "test_scenario/local_dataset_2.feather"
    payload = json.dumps({"dataset_id": blob_name, "query_vector": query_vector})
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()

        result = response.json()
        print(result)
        if result["nearest_neighbor"]["indices"] == [
            1849,
            684,
            1948,
            1366,
            1674,
            848,
            1551,
            913,
            1439,
            689,
        ]:
            return True
        else:
            raise Exception("The response does not match the expected format.")
    except requests.RequestException as e:
        return blob_name, f"Request failed: {str(e)}"


def send_request(blob_name):

    url = "http://localhost:5000/KNN_search"
    query_vector = [random.random() for _ in range(128)]
    payload = json.dumps({"dataset_id": blob_name, "query_vector": query_vector})
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        return blob_name, response.json()
    except requests.RequestException as e:
        return blob_name, f"Request failed: {str(e)}"


def distribute_requests(blob_names, total_requests):

    requests_per_blob = total_requests // len(blob_names)
    randomized_blobs = []
    for _ in range(requests_per_blob):
        randomized_blobs.extend(random.sample(blob_names, len(blob_names)))
    return randomized_blobs


def is_ssh_session():
    return (
        "SSH_CONNECTION" in os.environ
        or "SSH_CLIENT" in os.environ
        or "SSH_TTY" in os.environ
    )


def make_requests(bucket_test, total_requests=100):

    if is_ssh_session():
        print("Running over Google Cloud VM.")
        bucket_name = "ai-drive-psg-2024-us-central1"
        folder_name = "test"
    else:
        print("Running locally.")
        bucket_name = "ai-drive-psg-2024-us-central1-public"
        folder_name = "test_scenario"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ai-drive-psg-2024-local-sa.json"

    client = storage.Client(project="ai-drive-psg-2024")
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder_name if folder_name else "")

    if bucket_test == "small":
        modified_blob_names = [
            blob.name
            for blob in blobs
            if not blob.name.endswith("/") and blob.size < 10 * 1024 * 1024
        ]
    elif bucket_test == "bucket":
        modified_blob_names = [
            blob.name for blob in blobs if not blob.name.endswith("/")
        ]
    print(f"Total blobs: {len(modified_blob_names)}")

    total_requests = (total_requests // len(modified_blob_names)) * len(
        modified_blob_names
    )

    c_result = check_result()
    if c_result:
        print("knn search is correct")

    randomized_blobs = distribute_requests(modified_blob_names, total_requests)

    start_time = time.time()

    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_blob = {
            executor.submit(send_request, blob_name): blob_name
            for blob_name in randomized_blobs
        }
        for future in as_completed(future_to_blob):
            blob_name = future_to_blob[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"{blob_name} generated an exception: {exc}")

    end_time = time.time()
    total_time = end_time - start_time

    successful_requests = sum(1 for r in results if not isinstance(r[1], str))
    print(f"Successful requests: {successful_requests}/{len(results)}")
    print(f"Total time taken for requests: {total_time:.2f} seconds")
    print(f"Average time per request: {total_time/len(results):.4f} seconds")


if __name__ == "__main__":

    b = int(
        input(
            "Enter 0 for testing with small files or Enter 1 for testing entire bucket: "
        )
    )
    if b not in [0, 1]:
        raise ValueError(
            "Invalid input: Please enter 0 for testing small files or 1 for testing entire bucket"
        )

    if b == 0:
        bucket_test = "small"
    elif b == 1:
        bucket_test = "bucket"

    make_requests(bucket_test, 100)
