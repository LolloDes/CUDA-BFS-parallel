#include <iostream>
#include <vector>
#include <fstream>
#include <limits> // Per std::numeric_limits
#include <cmath>  // Per std::isnan
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

void csr_to_dense(std::ifstream &file, std::vector<std::vector<float>> &denseMatrix) {
    int numRows, numCols, numValues;
    file >> numRows >> numCols >> numValues;

    // Inizializza la matrice con NaN
    denseMatrix = std::vector<std::vector<float>>(numRows, std::vector<float>(numCols, std::numeric_limits<float>::quiet_NaN()));

    int row, col;
    float value;
    while (file >> row >> col >> value) {
        row--; // Convert to 0-based index
        col--; // Convert to 0-based index
        denseMatrix[row][col] = value;
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
            if (std::isnan(val)) {
                matrixFile << "- ";
            } else {
                matrixFile << val << " ";
            }
        }
        matrixFile << std::endl;
    }

    matrixFile.close();
}

__global__ void BFS_parallel(int source, Pair* currentFrontier, int* currentFrontierSize, Pair* nextFrontier, int* nextFrontierSize, float* denseMatrix, int numRows, int numCols) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= *currentFrontierSize) return;

    while (true) {
        Pair node = currentFrontier[tid];
        bool found = false;

        // Controllo per i < j
        if (node.first < node.second) {
            for (int j = 0; j < numCols; ++j) {
                if (node.first < j && !std::isnan(denseMatrix[node.first * numCols + j])) {
                    int index = atomicAdd(nextFrontierSize, 1);
                    nextFrontier[index] = {node.first, j};
                    found = true;
                    break; // Interrompe il ciclo dopo aver trovato il primo elemento
                }
            }
        } else if (node.first > node.second) {
            // Controllo per i > j
            for (int j = 0; j < numCols; ++j) {
                if (node.first > j && !std::isnan(denseMatrix[node.first * numCols + j])) {
                    int index = atomicAdd(nextFrontierSize, 1);
                    nextFrontier[index] = {node.first, j};
                    found = true;
                    break; // Interrompe il ciclo dopo aver trovato il primo elemento
                }
            }
        } else {
            // Controllo per i == j
            for (int j = 0; j < numCols; ++j) {
                if (node.first == j && !std::isnan(denseMatrix[node.first * numCols + j])) {
                    int index = atomicAdd(nextFrontierSize, 1);
                    nextFrontier[index] = {node.first, j};
                    found = true;
                    break; // Interrompe il ciclo dopo aver trovato il primo elemento
                }
            }
        }

        if (found && node.first == source && node.second == source) {
            break; // Interrompe il ciclo se il nodo source è stato trovato
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
            break; // Interrompe il ciclo se non ci sono più nodi da esplorare
        }
    }
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
    std::cout << "Matrice formata\n";
    file.close();

    int numRows = denseMatrix.size();
    int numCols = denseMatrix[0].size();
    float* d_denseMatrix;
    cudaMalloc(&d_denseMatrix, numRows * numCols * sizeof(float));
    cudaMemcpy(d_denseMatrix, denseMatrix.data(), numRows * numCols * sizeof(float), cudaMemcpyHostToDevice);

    // Inizializza currentFrontier con una singola coppia
    std::vector<Pair> hostCurrentFrontier = {{source, source}};
    Pair* d_currentFrontier;
    int* d_currentFrontierSize;
    cudaMalloc(&d_currentFrontier, hostCurrentFrontier.size() * sizeof(Pair));
    cudaMalloc(&d_currentFrontierSize, sizeof(int));
    cudaMemcpy(d_currentFrontier, hostCurrentFrontier.data(), hostCurrentFrontier.size() * sizeof(Pair), cudaMemcpyHostToDevice);
    int currentFrontierSize = hostCurrentFrontier.size();
    cudaMemcpy(d_currentFrontierSize, &currentFrontierSize, sizeof(int), cudaMemcpyHostToDevice);

    // Inizializza nextFrontier vuoto
    Pair* d_nextFrontier;
    int* d_nextFrontierSize;
    cudaMalloc(&d_nextFrontier, numRows * numCols * sizeof(Pair)); // Assumiamo che la dimensione massima sia numRows * numCols
    cudaMalloc(&d_nextFrontierSize, sizeof(int));
    int nextFrontierSize = 0;
    cudaMemcpy(d_nextFrontierSize, &nextFrontierSize, sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (currentFrontierSize + blockSize - 1) / blockSize;
    BFS_parallel<<<numBlocks, blockSize>>>(source, d_currentFrontier, d_currentFrontierSize, d_nextFrontier, d_nextFrontierSize, d_denseMatrix, numRows, numCols);

    // Copia i risultati dalla GPU all'host
    cudaMemcpy(&nextFrontierSize, d_nextFrontierSize, sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<Pair> hostNextFrontier(nextFrontierSize);
    cudaMemcpy(hostNextFrontier.data(), d_nextFrontier, nextFrontierSize * sizeof(Pair), cudaMemcpyDeviceToHost);

    // Stampa i risultati
    for (const auto& p : hostNextFrontier) {
        std::cout << "Next Frontier Pair: (" << p.first << ", " << p.second << ")\n";
    }

    // Genera il file della matrice
    generate_matrix_file(denseMatrix, "output_matrix.txt");
    std::cout << "File della matrice generato\n";
    cudaFree(d_denseMatrix);
    cudaFree(d_currentFrontier);
    cudaFree(d_currentFrontierSize);
    cudaFree(d_nextFrontier);
    cudaFree(d_nextFrontierSize);
    cudaDeviceReset();

    return 0;
}
