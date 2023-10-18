
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <ctime>
#include <float.h>
#include <math.h>
#include <tuple>

std::mt19937_64 generator(static_cast<unsigned long>(std::time(0)));
std::uniform_real_distribution<double> distribution(0.0, 10.0);
// quick fisher-yates courtesy of chatgpt
std::vector<int> fisherYatesShuffle(int n, int k) {
    std::vector<int> indices(n); // Array to hold indices [0, 1, ..., n-1]

    // Initialize the array with indices
    for (int i = 0; i < n; ++i) {
        indices[i] = i;
    }

    // Fisher-Yates Shuffle
    for (int i = 0; i < k; ++i) {
        std::uniform_int_distribution<int> distribution(i, n - 1); // Create a uniform distribution
        int j = distribution(generator); // Generate a random index

        // Swap indices[i] and indices[j]
        std::swap(indices[i], indices[j]);
    }

    // Resize the vector to keep only the first k shuffled indices
    indices.resize(k);

    return indices;
}

std::tuple<int, int> getCSVDimensions(const std::string& filePath) 
{
    std::ifstream file(filePath);
    if (!file.is_open()) 
    {
        std::cerr << "Could not open the file: " << filePath << std::endl;
        return {0, 0};
    }

    int rows = 0;
    int cols = 0;
    std::string line;

    while (std::getline(file, line)) 
    {
        rows++;
        std::stringstream lineStream(line);
        std::string cell;
        int tempCols = 0;
        while (std::getline(lineStream, cell, ',')) 
        {
            tempCols++;
        }
        if (rows == 1) 
        {
            cols = tempCols; // Assume all rows have the same number of columns
        }
    }

    return {rows, cols};
}

std::vector<double> readCSVTo1DArray(const std::string& filePath) 
{
    std::ifstream file(filePath);
    if (!file.is_open()) 
    {
        std::cerr << "Could not open the file: " << filePath << std::endl;
        return {};
    }

    std::vector<double> data;
    std::string line;
    while (std::getline(file, line)) 
    {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) 
        {
            data.push_back(std::stod(cell));
        }
    }

    return data;
}

__global__
void update(double* datapoints, double* centroids, int* assigned, int* counts, int num_points, int num_centroids, int dims)
{
    // for now i will only use one block, just use a bunch of threads. I will add to this later.
    int idx = threadIdx.x; // we have num_points threads for this kernel
    double themin = DBL_MAX;
    double dist;
    // now we just iterate striding by dims over the centroid array. then we do euclidean distance between datapoints[idx:idx+dims]
    for (int i = 0; i < num_centroids; i++)
    {
        // euclidean distance
        for (int j = 0; j < dims; j++)
        {
            dist += pow((datapoints[(idx*dims)+j]-centroids[(i*dims)+j]), 2);
        } 
        // sqrt(x) and x are both monotonically increasing, dist is positive, so i don't need to sqrt dist for comparison.
        if (dist < themin)
        {
            themin = dist;
            assigned[idx] = i;
        }
    }
    if (idx < num_centroids*dims)
    {
        centroids[idx] = 0;
    }
    for (int i = 0; i < dims; i++)
    {
        atomicAdd(&centroids[(assigned[idx]*dims)+i], datapoints[(idx*dims)+i]);
    }
    atomicAdd(&counts[assigned[idx]], 1);
    if (idx < num_centroids*dims) // keep in mind that num centroids*dims has to be lest than num_clusters
    {
        centroids[idx] /= counts[idx/dims];
        if (idx < num_centroids)
        {
            counts[idx] = 0;
        }
    }
    assigned[idx] = 0;
}

int main()
{
    const int N_CLUSTERS = 3;
    const int N_ITERATIONS = 100;
    const std::string path = "iris_edit.csv";
    std::tuple<int, int> pts_dims = getCSVDimensions(path);
    const int N_DATAPTS = std::get<0>(pts_dims);
    const int DIMS = std::get<1>(pts_dims);
    std::vector<double> datapoints = readCSVTo1DArray(path);
    std::vector<int> indices = fisherYatesShuffle(N_DATAPTS, N_CLUSTERS);
    std::vector<double> centroids;

    for (int i = 0; i < N_CLUSTERS; i++)
    {
        for (int j = 0; j < DIMS; j++)
        {
            centroids.push_back(datapoints[indices[i] + j]);
        }
    }

    double* dataptsGPU = 0;
    double* centroidsGPU = 0;
    int* assignedGPU = 0;
    int* countsGPU = 0;

    cudaMalloc(&dataptsGPU, N_DATAPTS * DIMS * sizeof(double));
    cudaMalloc(&centroidsGPU, N_CLUSTERS * DIMS * sizeof(double));
    cudaMalloc(&assignedGPU, N_DATAPTS * sizeof(int));
    cudaMalloc(&countsGPU, N_CLUSTERS * sizeof(int));

    cudaMemcpy(dataptsGPU, datapoints.data(), N_DATAPTS * DIMS * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(centroidsGPU, centroids.data(), N_CLUSTERS * DIMS * sizeof(double), cudaMemcpyHostToDevice);

    for (int i = 0; i < N_ITERATIONS; i++)
    {
        update <<<1, N_DATAPTS>>> (dataptsGPU, centroidsGPU, assignedGPU, countsGPU, N_DATAPTS, N_CLUSTERS, DIMS);
    }

    return 0;
}