#include "../includes/cuptipp.hxx"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

__global__ void vectoradd(size_t n, int* a, int* b, int* c) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n) {
		c[index] = a[index] + b[index];
	}
}

int main(void) {

	int N = 1 << 20;
	thrust::host_vector<int> A(N);
	thrust::host_vector<int> B(N);
	thrust::host_vector<int> C(N);

	for (int i = 0; i < N; i++) {
		A[i] = i;
		B[i] = i;
		C[i] = A[i] + B[i];
	}

	thrust::device_vector<int> a = A;
	thrust::device_vector<int> b = B;
	thrust::device_vector<int> c(N); 

	int blockDim = 128;
	int gridDim = (N + blockDim - 1) / blockDim;
	vectoradd <<< gridDim, blockDim >>> (N, a.data().get(), b.data().get(), c.data().get());

	int errors = 0;
	for (int i = 0; i < N; i++) {
		if (c[i] != C[i]) errors++;
	}
	printf("Simple vectoradd finished, number of errors = %d.\n", errors);

	return 0;
}