# BFS parallel algorithm GPU101
This code is an implementation of the Breadth First Search (BFS) algorithm using parallelism in CUDA by Lorenzo De Simone

### Report on the Functionality of the CUDA BFS Parallel Program

#### BFS algorithm

The Breadth-First Search (BFS) algorithm is a fundamental graph traversal algorithm used to explore nodes and edges of a graph. It starts at a given source node and explores all its neighboring nodes at the present depth level before moving on to nodes at the next depth level. This process continues until all nodes reachable from the source node have been explored.

The BFS algorithm works as follows:
1. **Initialization**: 
   - A queue is initialized and the source node is enqueued.
   - The distance to the source node is set to 0, and all other nodes are initialized with a distance of infinity (or a large value indicating they are unvisited).

2. **Exploration**:
   - The algorithm dequeues a node from the front of the queue.
   - For each neighboring node that has not been visited, it is marked as visited, its distance is updated, and it is enqueued.

3. **Termination**:
   - The process repeats until the queue is empty, meaning all reachable nodes have been explored.

In the context of this CUDA implementation, the BFS algorithm is parallelized to leverage the GPU's processing power. Each thread in the GPU processes a node, updating distances and managing the frontier of nodes to be explored next.

#### Introduction

This program implements a Breadth-First Search (BFS) algorithm using CUDA to leverage the parallel processing capabilities of a GPU. The primary goal is to calculate the distances from a source node to other nodes in a matrix, specifically along the diagonal. The program reads a matrix from a file, processes it using CUDA, and outputs the distances to a file.

#### Key Components

1. **Libraries and Definitions**:
   - The program includes standard libraries such as `<iostream>`, `<vector>`, `<fstream>`, `<limits>`, and `<cmath>`.
   - The CUDA runtime library `<cuda_runtime.h>` is included for GPU operations.
   - Macros for error checking (`CHECK` and `CHECK_KERNELCALL`) are defined to handle CUDA errors.

2. **Data Structures**:
   - A `Pair` struct is defined to hold two integers, `first` and `second`.

3. **Kernel Function**:
   - The `BFS_parallel` kernel function is the core of the parallel processing. It processes nodes in parallel to update their distances from the source node.

#### Workflow

1. **Reading the Matrix**:
   - The matrix is read from a file using the `csr_to_dense` function, which converts it into a dense matrix format.

2. **Memory Allocation**:
   - Memory is allocated on the GPU for the dense matrix, current frontier, next frontier, and distances using `cudaMalloc`.

3. **Initialization**:
   - The current frontier is initialized with the source node.
   - The distances vector is initialized with `-1` for all nodes, except the source node, which is set to `0`.

4. **Kernel Execution**:
   - The kernel `BFS_parallel` is launched with a specified number of blocks and threads per block.
   - The kernel processes nodes in parallel, updating their distances if they are along the diagonal.

5. **Timing**:
   - CUDA events are used to measure the execution time of the kernel. The start and stop events are recorded, and the elapsed time is calculated and printed.

6. **Copying Results**:
   - The results (distances) are copied from the GPU to the host memory using `cudaMemcpy`.

7. **Generating Output**:
   - The `generate_distance_file` function writes the distances to an output file, including the coordinates and distances of each node.

8. **Cleanup**:
   - GPU memory is freed using `cudaFree`, and the CUDA device is reset.