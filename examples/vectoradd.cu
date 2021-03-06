#include <cuptipp.hxx>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define PROFILE_ALL 1

__global__ void
vectorAdd(size_t n, int* a, int* b, int* c)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    c[index] = a[index] + b[index];
  }
}

int
main(int argc, char* argv[])
{
  int N = 1 << 19;
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

#if PROFILE_ALL
  int device = 0;
  std::vector<cuptipp::event_name_t> events = cuptipp::available_events(device);
#else
  std::vector<cuptipp::event_name_t> events{
    "active_warps",
    "atom_count",
    "l2_subp0_total_write_sector_queries",
    "l2_subp1_total_write_sector_queries",
    "l2_subp0_total_read_sector_queries",
    "l2_subp1_total_read_sector_queries"
  };
#endif

  cuptipp::profile profiler(events);
  profiler.begin();

  int blockDim = 128;
  int gridDim = (N + blockDim - 1) / blockDim;
  int iterations = profiler.get_iterations();

  cudaDeviceSynchronize();
  profiler.launch(iterations,
                  cuptipp::launch_t::PROFILED,
                  vectorAdd,
                  { gridDim, blockDim },
                  N,
                  a.data().get(),
                  b.data().get(),
                  c.data().get());
  cudaDeviceSynchronize();
  profiler.end();

  // Returns the kernel symbol-name
  std::vector<std::string> kernel_names = profiler.get_kernel_names();
  for (auto& k : kernel_names) {
    std::cout << "Kernel Name (symbol): " << k << "\n\n";
  }

  // Return event values for a given kernel names
  std::vector<cuptipp::event_value_t> event_values =
    profiler.get_event_values(kernel_names[0].c_str());
  for (int i = 0; i < event_values.size(); i++) {
    std::cout << "\t" << events[i] << ":\t" << event_values[i] << "\n";
  }

  /*
  {
    int errors = 0;
    for (int i = 0; i < N; i++) {
      if (c[i] != C[i])
        errors++;
    }
    printf("Simple vectoradd finished, number of errors = %d.\n", errors);
  }
  */

  return 0;
}