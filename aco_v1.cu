#include <stdio.h>

#define CITIES 734
#define BLOCK_DIM 32
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <helper_cuda.h>

__device__ float getRandom(uint64_t seed, int tid, int threadCallCount) {
    curandState s;
    curand_init(seed + tid + threadCallCount, 0, 0, &s);
    return curand_uniform(&s) + 0.00001f;
}


__global__ void
constructToursKernel(const float *pheromones, const float *heuristic, int *tours, int numCities, float alpha, float beta, uint64_t seed)
{
    int antId = blockDim.x * blockIdx.x + threadIdx.x;

	bool* visitedCities = (bool *)malloc(sizeof(bool) * numCities);
    float currentCityUncasted = getRandom(seed, antId, 0);
	int currentCity = currentCityUncasted * numCities;
	tours[antId * numCities] = currentCity;
	for (int i = 0; i < numCities; ++i) {
		visitedCities[i] = false;
	}
	visitedCities[currentCity] = true;
	for (int i = 1; i < numCities; ++i) {
		float* probabilities = (float *)malloc(sizeof(float) * numCities);
		float totalProbability = 0.0f;
		for (int j = 0; j < numCities; ++j) {
			if (visitedCities[j]) {
				probabilities[j] = 0.0f;
			} else {
				probabilities[j] = powf(pheromones[currentCity * numCities + j], alpha) * powf(heuristic[currentCity * numCities + j], beta);
				totalProbability += probabilities[j];		
			}
		}
		float randomValue = getRandom(seed, antId, 1) * totalProbability;
		float sum = 0.0f;
		for (int j = 0; j < numCities; ++j) {
			sum += probabilities[j];
			if (randomValue <= sum) {
				currentCity = j;
				break;
			}
		}
		free(probabilities);
		visitedCities[currentCity] = true;
		tours[antId * numCities + i] = currentCity;
	}

	free(visitedCities);
}

__global__ void
constructToursKernelDataParallelism(const float *pheromones, const float *heuristics, int *tours,  const int numCities, const float alpha, const float beta, uint64_t seed)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	__shared__ int indeces[BLOCK_DIM];
	__shared__ bool visited[CITIES];
	__shared__ float probabilities[BLOCK_DIM];
	__shared__ int currentCity[1];
	if (threadIdx.x == 0) {
		float currentCityUncasted = getRandom(seed, tid, 0);
		currentCity[0] = currentCityUncasted * numCities;
		for (int i = 0; i < numCities; ++i) {
			visited[i] = false;
		}
		visited[currentCity[0]] = true;
		tours[blockIdx.x * numCities] = currentCity[0];
	}
	__syncthreads();
	int citiesPerThread = numCities / blockDim.x;
	int threadsWithAdditionalCity = numCities % blockDim.x;
	int selectedCity = threadIdx.x;
	for (int i = 1; i < numCities; ++i) {
		float maxHeuristic = 0.0f;
		if (!visited[selectedCity]) {
			maxHeuristic = heuristics[currentCity[0] * numCities + selectedCity];
		}
		for (int j = 0; j < citiesPerThread; ++j) {
			int consideredCity = j * BLOCK_DIM + threadIdx.x;
			if (heuristics[currentCity[0] * numCities + consideredCity] > maxHeuristic && !visited[consideredCity]) {
				selectedCity = consideredCity;
				maxHeuristic = heuristics[currentCity[0] * numCities + consideredCity];
			}
		}
		if (threadIdx.x < threadsWithAdditionalCity) {
			int consideredCity = citiesPerThread * BLOCK_DIM + threadIdx.x;
			if (heuristics[currentCity[0] * numCities + consideredCity] >= maxHeuristic && !visited[consideredCity]) {
				selectedCity = consideredCity;
				maxHeuristic = heuristics[currentCity[0] * numCities + selectedCity];
			}
		}
		__syncthreads();
		probabilities[threadIdx.x] = powf(pheromones[currentCity[0] * numCities + selectedCity], alpha) * powf(maxHeuristic, beta) * getRandom(seed, tid, i);	
		indeces[threadIdx.x] = selectedCity;

		__syncthreads();

		for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
			if (threadIdx.x < stride) {
				if (probabilities[threadIdx.x] < probabilities[threadIdx.x + stride]) {
					probabilities[threadIdx.x] = probabilities[threadIdx.x + stride];
					indeces[threadIdx.x] = indeces[threadIdx.x + stride];
				}
			}
			__syncthreads();
		}

		if (threadIdx.x == 0) {
			currentCity[0] = indeces[0];
			tours[blockIdx.x * numCities + i] = currentCity[0];
			visited[currentCity[0]] = true;
		}

		__syncthreads();
	}
}

__global__ void
findBestTourKernel(float *lengths, int *bestTourIndex) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	__shared__ int indeces[CITIES];
	indeces[threadIdx.x] = threadIdx.x;
	__syncthreads();
	for (int stride = CITIES / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride) {
			if (lengths[threadIdx.x] < lengths[threadIdx.x + stride]) {
				lengths[threadIdx.x] = lengths[threadIdx.x + stride];
				indeces[threadIdx.x] = indeces[threadIdx.x + stride];
				lengths[threadIdx.x] = lengths[threadIdx.x + stride];
				indeces[threadIdx.x] = indeces[threadIdx.x + stride];
			}
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		bestTourIndex[0] = indeces[0];
	}
}

__global__ void
pheromoneEvaporationKernel(float *pheromones, float evaporationRate)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

	pheromones[i] *= evaporationRate;
	if (pheromones[i] < 0.1) {
		pheromones[i] = 0.1;
	}
}

__global__ void
updatePheromonesKernel(int* bestTour, float *pheromones, const float length, const int numCities)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i != (numCities - 1)) {
		pheromones[bestTour[i] * numCities + bestTour[i + 1]] += (1000.0f / length);
		if (pheromones[bestTour[i] * numCities + bestTour[i + 1]] > 0.9f) {
			pheromones[bestTour[i] * numCities + bestTour[i + 1]] = 0.9f;
		}
	} else {
		pheromones[bestTour[0] * numCities + bestTour[numCities - 1]] += (1000.0f / length);
		if (pheromones[bestTour[0] * numCities + bestTour[numCities - 1]] > 0.9f) {
			pheromones[bestTour[0] * numCities + bestTour[numCities - 1]] = 0.9f;
		}
	}
}

__global__ void
constructBestTourKernel(int *tours, int* bestTour, int* bestTourIndex)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

	bestTour[i] = tours[bestTourIndex[0] * CITIES + i];
}

float distance(int x1, int y1, int x2, int y2) {
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

__global__ void
findToursLengthsKernel(float *lengths, float *heuristics, int *tours) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	for (int j = 0; j < CITIES - 1; ++j) {
		lengths[i] += 1.0 / heuristics[tours[i * CITIES + j] * CITIES + tours[i * CITIES + j + 1]]; 
	}
	lengths[i] += 1.0 / heuristics[tours[i * CITIES + CITIES - 1] * CITIES + tours[i * CITIES + 0]]; 
}

/**
 * Host main routine
 */
int
main(void)
{
	const int size = 734;
	float coordinates[size][2];
    float distanceMatrix[size][size];

	FILE *file = fopen("path", "r");
    for (int i = 0; i < size; i++) {
		fscanf(file, "%*d %f %f", &coordinates[i][0], &coordinates[i][1]);
    }
    fclose(file);
    for (int i = 0; i < size; i++) {
        for (int j = i; j < size; j++) {
            if (i == j) {
                distanceMatrix[i][j] = 0;
            } else {
                float dist = distance(coordinates[i][0], coordinates[i][1], coordinates[j][0], coordinates[j][1]);
                distanceMatrix[i][j] = dist;
                distanceMatrix[j][i] = dist;
            }
        }
    }

    cudaError_t err = cudaSuccess;

    int numCities = 734;
	int threadsPerBlock = 32;
    int blocksPerGrid = 256;

    float *h_pheromones = (float *)malloc(sizeof(float) * numCities * numCities);
    float *h_heuristic = (float *)malloc(sizeof(float) * numCities * numCities);
	int *h_tours = (int *)malloc(sizeof(int) * numCities * blocksPerGrid);
	float *h_lengths = (float *)malloc(sizeof(float) * blocksPerGrid);
	int *h_bestTourIndex = (int *)malloc(sizeof(int));
	int *h_bestTour = (int *)malloc(sizeof(int) * numCities);

    if (h_pheromones == NULL || h_heuristic == NULL || h_tours == NULL || h_lengths == NULL || h_bestTour == NULL || h_bestTourIndex == NULL) {
        fprintf(stderr, "Failed to allocate host arrays!\n");
        exit(EXIT_FAILURE);
    }

	h_bestTourIndex[0] = 0;

    for (int i = 0; i < numCities * numCities; ++i) {
        h_pheromones[i] = 0.9f;
    }

    for (int i = 0; i < blocksPerGrid; ++i) {
        h_lengths[i] = 0.0f;
    }

    for (int i = 0; i < numCities * numCities; ++i) {
		int a = i / numCities;
		int b = i % numCities;
		if (a == b) {
			h_heuristic[i] = 0.0f;
		} else {	
			h_heuristic[i] = 1.0f / (float) distanceMatrix[a][b];
		}
    }

	for (int i = 0; i < numCities * blocksPerGrid; ++i) {
        h_tours[i] = 0;
    }

    float *d_pheromones = NULL;
    err = cudaMalloc((void **)&d_pheromones, sizeof(float) * numCities * numCities);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device pheromones array (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_lengths = NULL;
    err = cudaMalloc((void **)&d_lengths, sizeof(float) * blocksPerGrid);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device pheromones array (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	int *d_bestTourIndex = NULL;
    err = cudaMalloc((void **)&d_bestTourIndex, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device best tour index (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_heuristic = NULL;
    err = cudaMalloc((void **)&d_heuristic, sizeof(float) * numCities * numCities);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device heuristic array (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int *d_tours = NULL;
    err = cudaMalloc((void **)&d_tours, sizeof(int) * numCities * blocksPerGrid);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device tours array (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int *d_bestTour = NULL;
    err = cudaMalloc((void **)&d_bestTour, sizeof(int) * numCities);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device best tour array (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
    float *d_distanceMatrix = NULL;
    err = cudaMalloc((void **)&d_distanceMatrix, sizeof(float) * numCities * numCities);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device best tour array (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	float alpha = 4.0f; 
	float beta = 1.5f;
	float evaporationRate = 0.7f;

    printf("Copy input data from the host memory to the CUDA device\n");
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
    err = cudaMemcpy(d_pheromones, h_pheromones, sizeof(float) * numCities * numCities, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy pheromones array from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	err = cudaMemcpy(d_heuristic, h_heuristic, sizeof(float) * numCities * numCities, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy heuristic array from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	err = cudaMemcpy(d_tours, h_tours, sizeof(int) *  numCities * blocksPerGrid, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy tours array from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	err = cudaMemcpy(d_lengths, h_lengths, sizeof(float) * blocksPerGrid, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy lengths array from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	for(int i = 0; i < 10; ++i) {
		cudaMemcpy(d_pheromones, h_pheromones, sizeof(float) * numCities * numCities, cudaMemcpyHostToDevice);
		cudaMemcpy(d_heuristic, h_heuristic, sizeof(float) * numCities * numCities, cudaMemcpyHostToDevice);
		cudaMemcpy(d_tours, h_tours, sizeof(int) * blocksPerGrid * numCities, cudaMemcpyHostToDevice);
		constructToursKernelDataParallelism<<<blocksPerGrid, threadsPerBlock>>>(d_pheromones, d_heuristic, d_tours, numCities, alpha, beta, time(NULL));
		// constructToursKernel<<<1, blocksPerGrid>>>(d_pheromones, d_heuristic, d_tours, numCities, alpha, beta, time(NULL));
		cudaMemcpy(h_tours, d_tours, sizeof(int) * blocksPerGrid * numCities, cudaMemcpyDeviceToHost);
		cudaMemcpy(d_tours, h_tours, sizeof(int) * blocksPerGrid * numCities, cudaMemcpyHostToDevice);
		cudaMemcpy(d_lengths, h_lengths, sizeof(float) * blocksPerGrid, cudaMemcpyHostToDevice);
		cudaMemcpy(d_heuristic, h_heuristic, sizeof(float) * numCities * numCities, cudaMemcpyHostToDevice);
		findToursLengthsKernel<<<1, blocksPerGrid>>>(d_lengths, d_heuristic, d_tours);
		cudaMemcpy(h_lengths, d_lengths, sizeof(float) * blocksPerGrid, cudaMemcpyDeviceToHost);
		pheromoneEvaporationKernel<<<numCities, numCities>>>(d_pheromones, evaporationRate);
		cudaMemcpy(h_pheromones, d_pheromones, sizeof(float) * numCities * numCities, cudaMemcpyDeviceToHost);
		cudaMemcpy(d_bestTourIndex, h_bestTourIndex, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_lengths, h_lengths, sizeof(float) * blocksPerGrid, cudaMemcpyHostToDevice);
		findBestTourKernel<<<1, numCities>>>(d_lengths, d_bestTourIndex);
		cudaMemcpy(h_bestTourIndex, d_bestTourIndex, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(d_tours, h_tours, sizeof(int) * blocksPerGrid * numCities, cudaMemcpyHostToDevice);
		cudaMemcpy(d_bestTour, h_bestTour, sizeof(int) * numCities, cudaMemcpyHostToDevice);
		cudaMemcpy(d_bestTourIndex, h_bestTourIndex, sizeof(int), cudaMemcpyHostToDevice);
		constructBestTourKernel<<<1, numCities>>>(d_tours, d_bestTour, d_bestTourIndex);
		cudaMemcpy(h_bestTour, d_bestTour, sizeof(int) * numCities, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_bestTourIndex, d_bestTourIndex, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(d_bestTour, h_bestTour, sizeof(int) * numCities, cudaMemcpyHostToDevice);
		cudaMemcpy(d_pheromones, h_pheromones, sizeof(float) * numCities * numCities, cudaMemcpyHostToDevice);
		updatePheromonesKernel<<<1, numCities>>>(d_bestTour, d_pheromones, h_lengths[h_bestTourIndex[0]], numCities);
		cudaMemcpy(h_bestTour, d_bestTour, sizeof(int) * numCities, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_pheromones, d_pheromones, sizeof(float) * numCities * numCities, cudaMemcpyDeviceToHost);
	}

	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float milliseconds;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%f", milliseconds);

    err = cudaFree(d_pheromones);
    err = cudaFree(d_heuristic);
	err = cudaFree(d_tours);

	float *lengths = (float *)malloc(sizeof(float) * blocksPerGrid);
	for (int j = 0; j < blocksPerGrid; ++j) {
		lengths[j] = 0.0f;
	}
	for (int j = 0; j < blocksPerGrid; ++j) {
		for (int k = 0; k < numCities - 1; ++k) {
			lengths[j] += 1.0f / h_heuristic[h_tours[j * numCities + k] * numCities + h_tours[j * numCities + k + 1]];
		}
		lengths[j] += distanceMatrix[h_tours[j * numCities + numCities - 1]][h_tours[j * numCities + 0]];
	}
	int c, location = 0;
	for (c = 1; c < blocksPerGrid; c++)
		if (lengths[c] < lengths[location])
			location = c;
	printf("\n");
	printf("%f %d", lengths[location], location);
	printf("\n");
	int *bestTour = (int *)malloc(sizeof(int) * numCities);
	for (int j = 0; j < numCities; ++j) {
		bestTour[j] = h_tours[location * numCities + j];
		printf("%d, ", bestTour[j]);
	}

    free(h_pheromones);
    free(h_heuristic);
	free(h_tours);

    printf("Done\n");
    return 0;
}
