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
__global__ void BFS_parallel(int source,std::vector<std::vector<float>> denseMatrix, Pair* currentfrontier, Pair* nextfrontier){
  /*

  1. leggere primo elemento
  2. controllare elementi vicini
  3. aggiungere al vettore livello
  while(true){
    Pair node = currentFrontier[tid];
    bool found=false;
    if (node.first < node.second) {
    // Controllo per i < j
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
          break; // Interrompe il ciclo dopo aver trovato il primo elemento
          }
        }
      }
  }
    4. ripetere in parallelo
      currentfrontier=nextfrontier;
      nextfrontier.clear();
    }
    if (found && node.first == source && node.second == source)
      break; // Interrompe il ciclo se il nodo source è stato trovato
  }
  5. vettore distanza
   5.1. col=row diminuisce ad aumentare di row e col
   5.2. controllo se NaN
   5.3. NaN->passo al prossimo
   5.4. Non NaN->
    5.4.1. scrivo numero nodo, 
    5.4.2. aumento distanza, 
    5.4.3. scrivo distanza
  6. stampa vettore distanze in un file*/
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
