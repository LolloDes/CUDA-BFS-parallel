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
#include <limits>
#include <numeric>
#include <cmath>  // Per std::isnan

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
    file.close();

    generate_matrix_file(denseMatrix, "dense_matrix.txt");

    return 0;
}
