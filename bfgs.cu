#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <time.h>
#include <vector>
#include <sstream>
#include <algorithm>

#define MAX_FRONTIER_SIZE 128

#define CHECK(call)                                                                 \
  {                                                                                 \
    const cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                       \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }

#define CHECK_KERNELCALL()                                                          \
  {                                                                                 \
    const cudaError_t err = cudaGetLastError();                                     \
    if (err != cudaSuccess) {                                                       \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }

// Kernel to perform parallel BFS with 3 threads
__global__ void BFS_parallel(const int *rowPointers, const int *destinations, int *currentFrontier, int *nextFrontier, int *currentFrontierSize, int *nextFrontierSize, int *distances, int numVertices) {
    int tid = threadIdx.x;
    printf("Thread %d starting\n", tid);

    while (*currentFrontierSize > 0) {
        if (tid == 0) {
            printf("Current frontier size: %d\n", *currentFrontierSize);
        }
        __syncthreads();

        for (int f = tid; f < *currentFrontierSize; f += blockDim.x) {
            const int currentVertex = currentFrontier[f];
            printf("Thread %d processing vertex %d\n", tid, currentVertex);
            for (int i = rowPointers[currentVertex]; i < rowPointers[currentVertex + 1]; ++i) {
                const int neighbor = destinations[i];
                printf("Thread %d examining neighbor %d\n", tid, neighbor);
                if (atomicCAS(&distances[neighbor], -1, distances[currentVertex] + 1) == -1) {
                    int index = atomicAdd(nextFrontierSize, 1);
                    nextFrontier[index] = neighbor;
                    printf("Thread %d added neighbor %d to next frontier\n", tid, neighbor);
                }
            }
        }
        __syncthreads();

        if (tid == 0) {
            int *temp = currentFrontier;
            currentFrontier = nextFrontier;
            nextFrontier = temp;
            *currentFrontierSize = *nextFrontierSize;
            *nextFrontierSize = 0;
            printf("Swapped frontiers, new size: %d\n", *currentFrontierSize);
        }
        __syncthreads();
    }
}

void generate_matrix_file(const std::vector<std::vector<float>> &matrix, const std::string &filename) {
    std::ofstream matrixFile(filename);
    if (!matrixFile.is_open()) {
        std::cerr << "Cannot open matrix file!\n";
        return;
    }

    matrixFile << "Dense Matrix:" << std::endl;
    for (const auto &row : matrix) {
        for (const auto &val : row) {
            matrixFile << val << " ";
        }
        matrixFile << std::endl;
    }

    matrixFile.close();
}

void generate_distances_file(const std::vector<int> &distances, const std::string &filename) {
    std::ofstream distFile(filename);
    if (!distFile.is_open()) {
        std::cerr << "Cannot open distances file!\n";
        return;
    }

    distFile << "Distances:" << std::endl;
    for (size_t i = 0; i < distances.size(); ++i) {
        distFile << "Node " << i << ": " << distances[i] << std::endl;
    }

    distFile.close();
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cout << "Usage: ./exec matrix_file source\n";
        return 0;
    }

    std::string filename = argv[1];
    int source = atoi(argv[2]);

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "File cannot be opened!\n";
        return 1;
    }

    int numRows, numCols, numValues;
    file >> numRows >> numCols >> numValues;

    std::vector<int> rowPointers(numRows + 1, 0);
    std::vector<int> destinations;
    std::vector<float> values;

    int row, col;
    float value;
    while (file >> row >> col >> value) {
        row--; // Convert to 0-based index
        col--; // Convert to 0-based index
        rowPointers[row + 1]++;
        destinations.push_back(col);
        values.push_back(value);
    }

    // Convert rowPointers to cumulative sum
    for (int i = 0; i < numRows; ++i) {
        rowPointers[i + 1] += rowPointers[i];
    }

    file.close();

    // Create a dense matrix
    std::vector<std::vector<float>> denseMatrix(numRows, std::vector<float>(numCols, 0.0f));
    for (int i = 0; i < numRows; ++i) {
        for (int j = rowPointers[i]; j < rowPointers[i + 1]; ++j) {
            denseMatrix[i][destinations[j]] = values[j];
        }
    }

    // Print the dense matrix to the terminal
    std::cout << "Dense Matrix:" << std::endl;
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            std::cout << denseMatrix[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Generate matrix file
    generate_matrix_file(denseMatrix, "dense_matrix.txt");

    // Allocate memory for frontiers and distances
    int *d_rowPointers, *d_destinations, *d_currentFrontier, *d_nextFrontier, *d_currentFrontierSize, *d_nextFrontierSize, *d_distances;
    CHECK(cudaMalloc(&d_rowPointers, (numRows + 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_destinations, destinations.size() * sizeof(int)));
    CHECK(cudaMalloc(&d_currentFrontier, numRows * sizeof(int)));
    CHECK(cudaMalloc(&d_nextFrontier, numRows * sizeof(int)));
    CHECK(cudaMalloc(&d_currentFrontierSize, sizeof(int)));
    CHECK(cudaMalloc(&d_nextFrontierSize, sizeof(int)));
    CHECK(cudaMalloc(&d_distances, numRows * sizeof(int)));

    CHECK(cudaMemcpy(d_rowPointers, rowPointers.data(), (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_destinations, destinations.data(), destinations.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_currentFrontier, -1, numRows * sizeof(int)));
    CHECK(cudaMemset(d_nextFrontier, -1, numRows * sizeof(int)));
    CHECK(cudaMemset(d_distances, -1, numRows * sizeof(int)));

    int blockSize = 3; // Use 3 threads per block
    int numBlocks = 1; // Use 1 block
    std::cout << "Number of blocks: " << numBlocks << std::endl;

    // Initialize the frontier with the source node
    int h_currentFrontierSize = 1;
    int h_nextFrontierSize = 0;
    CHECK(cudaMemcpy(d_currentFrontierSize, &h_currentFrontierSize, sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_nextFrontierSize, &h_nextFrontierSize, sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_currentFrontier, &source, sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_distances + source, 0, sizeof(int))); // Initialize source distance to 0
    std::cout << "Source node: " << source << std::endl;

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    std::cout << "Starting BFS..." << std::endl;

    // Debug: Print parameters before kernel call
    std::cout << "Calling BFS_parallel with parameters:" << std::endl;
    std::cout << "numBlocks: " << numBlocks << ", blockSize: " << blockSize << std::endl;
    std::cout << "d_rowPointers: " << d_rowPointers << ", d_destinations: " << d_destinations << std::endl;
    std::cout << "d_currentFrontier: " << d_currentFrontier << ", d_nextFrontier: " << d_nextFrontier << std::endl;
    std::cout << "d_currentFrontierSize: " << d_currentFrontierSize << ", d_nextFrontierSize: " << d_nextFrontierSize << std::endl;
    std::cout << "d_distances: " << d_distances << std::endl;

    // Verify memory allocation
    if (d_rowPointers == nullptr || d_destinations == nullptr || d_currentFrontier == nullptr || d_nextFrontier == nullptr || d_currentFrontierSize == nullptr || d_nextFrontierSize == nullptr || d_distances == nullptr) {
        std::cerr << "Memory allocation failed!" << std::endl;
        return 1;
    }

    BFS_parallel<<<numBlocks, blockSize>>>(d_rowPointers, d_destinations, d_currentFrontier, d_nextFrontier, d_currentFrontierSize, d_nextFrontierSize, d_distances, numRows);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());
    std::cout << "BFS completed!" << std::endl;

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Elapsed time: " << milliseconds << " ms" << std::endl;

    // Copy distances back to host
    std::vector<int> h_distances(numRows);
    CHECK(cudaMemcpy(h_distances.data(), d_distances, numRows * sizeof(int), cudaMemcpyDeviceToHost));

    // Debug: Print distances to terminal
    std::cout << "Distances:" << std::endl;
    for (int i = 0; i < numRows; ++i) {
        std::cout << "Node " << i << ": " << h_distances[i] << std::endl;
    }

    // Generate distances file
    generate_distances_file(h_distances, "distances.txt");

    CHECK(cudaFree(d_rowPointers));
    CHECK(cudaFree(d_destinations));
    CHECK(cudaFree(d_currentFrontier));
    CHECK(cudaFree(d_nextFrontier));
    CHECK(cudaFree(d_currentFrontierSize));
    CHECK(cudaFree(d_nextFrontierSize));
    CHECK(cudaFree(d_distances));

    return 0;
}
