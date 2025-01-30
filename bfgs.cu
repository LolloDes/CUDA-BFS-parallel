#include <iostream>
#include <vector>
#include <fstream>
#include <limits> // For using std::numeric_limits
#include <cmath>  // For using std::isnan
#include <cuda_runtime.h>

#define MAX_FRONTIER_SIZE 128
struct Pair {
    int first;
    int second;
};
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
// Function to convert csr to dense matrix
void csr_to_dense(std::ifstream &file, std::vector<std::vector<float>> &denseMatrix) {
    int numRows, numCols, numValues;
    file >> numRows >> numCols >> numValues;

    // Initialize the dense matrix with NaN values
    denseMatrix = std::vector<std::vector<float>>(numRows, std::vector<float>(numCols, std::numeric_limits<float>::quiet_NaN()));

    int row, col;
    float value;
    while (file >> row >> col >> value) {
        row--; // Convert to 0-based index
        col--; // Convert to 0-based index
        denseMatrix[row][col] = value;
    }
}
    // Function to generate matrix file
void generate_matrix_file(const std::vector<std::vector<float>> &matrix, const std::string &filename) {
    std::ofstream matrixFile(filename);
    if (!matrixFile.is_open()) {
        std::cerr << "Cannot open matrix file!\n";
        return;
    }

    matrixFile << "Dense Matrix:" << std::endl;
    for (const auto &row : matrix) {
        for (const auto &val : row) {
            if (std::isnan(val)) {
                matrixFile << "- ";
            } else {
                matrixFile << val << " ";
            }
        }
        matrixFile << std::endl;
    }
    std::cout << "Matrix file generated\n";

    matrixFile.close();
}

__global__ void BFS_parallel(int source, Pair* currentFrontier, int* currentFrontierSize, Pair* nextFrontier, int* nextFrontierSize, float* denseMatrix, int numRows, int numCols, int* distances) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= *currentFrontierSize) return;
    int d=0;
    while (true) {
        Pair node = currentFrontier[tid];
        bool found = false;
        // Check element along the column (i<j)
        if (node.first < node.second) {
            for (int j = 0; j < numCols; ++j) {
                if (node.first < j && !std::isnan(denseMatrix[node.first * numCols + j])) {
                    int index = atomicAdd(nextFrontierSize, 1);
                    nextFrontier[index] = {node.first, j};
                    found = true;
                    break; // Breaks the loop after finding the first element
                }
            }
        } else if (node.first > node.second) {
            // Check element along the row (i>j) 
            for (int j = 0; j < numRows; ++j) {
                if (node.first > j && !std::isnan(denseMatrix[node.first * numCols + j])) {
                    int index = atomicAdd(nextFrontierSize, 1);
                    nextFrontier[index] = {node.first, j};
                    found = true;
                    break; // Breaks the loop after finding the first element
                }
            }
        } else {
            // Check element along the diagonal (i=j)
            for (int j = 0; j < numCols; ++j) {
                if (node.first == j && !std::isnan(denseMatrix[node.first * numCols + j])) {
                    int index = atomicAdd(nextFrontierSize, 1);
                    nextFrontier[index] = {node.first, j};
                    found = true;
                    distances[d] = node.first; // Aggiorna la distanza
                    d++;
                    break; // Breaks the loop after finding the first element
                }
            }
        }

        if (found && node.first == source && node.second == source) {
            break; // Breaks the loop if the source node is found
        }

        __syncthreads();

        if (tid == 0) {
            *currentFrontierSize = *nextFrontierSize;
            *nextFrontierSize = 0;
            Pair* temp = currentFrontier;
            currentFrontier = nextFrontier;
            nextFrontier = temp;
        }

        __syncthreads();

        if (*currentFrontierSize == 0) {
            break; // Stops the cycle if there are no more nodes to explore
        }
    }
}

void generate_distance_file(const std::vector<int> &distances, int source, int numRows, int numCols, const std::string &filename) {
    std::ofstream distanceFile(filename);
    if (!distanceFile.is_open()) {
        std::cerr << "Cannot open distance file!\n";
        return;
    }

    distanceFile << "Distances:" << std::endl;
    for (int i = 0; i < source; ++i) {
            int dist=source-i-1;
            distanceFile << "Node (" << distances[i] << ", " << distances[i] << "): " << dist;
            if (i != (source)) {
                distanceFile <<std::endl;
            } else {
                distanceFile << " (source)" << std::endl;
           }
           
    }
    distanceFile.close();
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

    std::vector<std::vector<float>> denseMatrix;
    csr_to_dense(file, denseMatrix);
    std::cout << "Matrix formed\n";
    generate_matrix_file(denseMatrix, "matrix.txt");
    file.close();

    int numRows = denseMatrix.size();
    int numCols = denseMatrix[0].size();
    if (source>numCols){
        std::cerr << "Source node is not in the matrix!\n";
        return 1;}
    float* d_denseMatrix;
    cudaMalloc(&d_denseMatrix, numRows * numCols * sizeof(float));
    cudaMemcpy(d_denseMatrix, denseMatrix.data(), numRows * numCols * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize currentFrontier with a single pair
    std::vector<Pair> hostCurrentFrontier = {{0, 0}};
    Pair* d_currentFrontier;
    int* d_currentFrontierSize;
    cudaMalloc(&d_currentFrontier, hostCurrentFrontier.size() * sizeof(Pair));
    cudaMalloc(&d_currentFrontierSize, sizeof(int));
    cudaMemcpy(d_currentFrontier, hostCurrentFrontier.data(), hostCurrentFrontier.size() * sizeof(Pair), cudaMemcpyHostToDevice);
    int currentFrontierSize = hostCurrentFrontier.size();
    cudaMemcpy(d_currentFrontierSize, &currentFrontierSize, sizeof(int), cudaMemcpyHostToDevice);

    // Initialize nextFrontier empty
    Pair* d_nextFrontier;
    int* d_nextFrontierSize;
    cudaMalloc(&d_nextFrontier, numRows * numCols * sizeof(Pair)); // Let's assume the maximum size is numRows * numCols
    cudaMalloc(&d_nextFrontierSize, sizeof(int));
    int nextFrontierSize = 0;
    cudaMemcpy(d_nextFrontierSize, &nextFrontierSize, sizeof(int), cudaMemcpyHostToDevice);

    // Initialize distance vector
    std::vector<int> hostDistances(numRows, -1 ); // Initialize all distances to -1
    hostDistances[(source - 1)] = source; // The distance of the source node is 0
    int* d_distances;
    cudaMalloc(&d_distances, sizeof(int));
    cudaMemcpy(d_distances, hostDistances.data(), sizeof(int), cudaMemcpyHostToDevice);
    
    // Timing variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start the timer
    std::cout << "Starting kernel execution\n";
    cudaEventRecord(start);

    int blockSize = 256;
    int numBlocks = (currentFrontierSize + blockSize - 1) / blockSize;
    BFS_parallel<<<numBlocks, blockSize>>>(source, d_currentFrontier, d_currentFrontierSize, d_nextFrontier, d_nextFrontierSize, d_denseMatrix, numRows, numCols, d_distances);

    // Stop the timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate the execution time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;
    // Copy results from GPU to host
    cudaMemcpy(hostDistances.data(), d_distances, sizeof(int), cudaMemcpyDeviceToHost);

    // Generate distance file
    generate_distance_file(hostDistances, source, numRows, numCols, "distances.txt");
    std::cout << "Distance file generated\n";
    cudaFree(d_denseMatrix);
    cudaFree(d_currentFrontier);
    cudaFree(d_currentFrontierSize);
    cudaFree(d_nextFrontier);
    cudaFree(d_nextFrontierSize);
    cudaFree(d_distances);
    cudaDeviceReset();

    return 0;
}
