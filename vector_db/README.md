# KNN Search API

## Overview

This project implements a K-Nearest Neighbors (KNN) search API using Flask and Waitress, with the dataset stored in Google Cloud Storage (GCS). The API allows users to upload a query vector and retrieve the nearest neighbors from a dataset of embeddings.

## Optimizations Implemented

### 1. Caching with Threading

- **Issue**: Loading the dataset from GCS for every request is time-consuming.
- **Solution**: Implemented a global cache with threading locks to load the dataset into memory once and reuse it for subsequent requests.

### 2. Efficient Dot Product Calculation

- **Issue**: Calculating dot products can be computationally expensive.
- **Solution**: Used `numpy` for efficient vectorized operations to calculate the dot product and find the top K nearest neighbors.

### 3. Multi-threaded Server with Waitress

- **Issue**: Handling multiple requests efficiently.
- **Solution**: Used `waitress` with multiple threads to handle concurrent requests.

### 4. Reduced I/O Overhead

- **Issue**: Frequent I/O operations can slow down the API.
- **Solution**: Cached the dataset in memory to minimize I/O operations.

### 5. Environment-specific Configurations

- **Issue**: Different environments (local vs. cloud) require different configurations.
- **Solution**: Used environment variables to handle configurations dynamically.

## What I Learned

1. **Concurrency in Python**: Implemented threading to manage shared resources effectively and handle concurrent requests.
2. **Efficient Data Processing**: Leveraged `numpy` for high-performance data operations.
3. **Cloud Storage Integration**: Integrated with Google Cloud Storage for scalable data storage solutions.
4. **API Development**: Built and optimized a RESTful API using Flask and Waitress.
5. **Performance Monitoring**: Identified and mitigated bottlenecks in the application.

## Potential Optimizations

1. **Asynchronous Processing**: Implement asynchronous request handling using `FASTapi` or similar frameworks to improve throughput.
2. **Load Balancing**: Implement load balancing with Nginx or another load balancer to distribute requests across multiple instances.
3. **Advanced Caching**: Use Redis or another in-memory datastore for caching frequently accessed data.
4. **Batch Processing**: Implement batch processing for handling multiple queries in a single request.
5. **Auto-scaling**: Deploy the application on a platform that supports auto-scaling to handle varying loads dynamically.
6. **Data Partitioning**: Partition the dataset to reduce the size of data each instance needs to handle, improving response times.

## Running the Server with Waitress

To run the server with Waitress using the specified host, port, and thread settings, execute the following:

```sh
waitress-serve --host=0.0.0.0 --port=5000 --threads=20 app:app
```
