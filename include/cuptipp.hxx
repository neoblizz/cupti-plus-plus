#pragma once

// C/C++
#include <cstddef>
#include <iostream>
#include <map>
#include <stdio.h>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

// CUDA Profiling Tools Interface (CUPTI)
// #include <cuda_runtime_api.h>
#include <cupti.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

#define CU_ERROR(error)                                                        \
  do {                                                                         \
    CUresult err = error;                                                      \
    if (error != CUDA_SUCCESS) {                                               \
      const char* errstr;                                                      \
      cuGetErrorString(err, &errstr);                                          \
      fprintf(stderr, "Error: %s %s %d\n", errstr, __FILE__, __LINE__);        \
      if (true)                                                                \
        exit(err);                                                             \
    }                                                                          \
  } while (0)

#define CUDA_ERROR(error)                                                      \
  do {                                                                         \
    cudaError_t err = error;                                                   \
    if (error != cudaSuccess) {                                                \
      const char* errstr = cudaGetErrorString(err);                            \
      fprintf(stderr, "Error: %s %s %d\n", errstr, __FILE__, __LINE__);        \
      if (true)                                                                \
        exit(err);                                                             \
    }                                                                          \
  } while (0)

#define CUPTI_ERROR(error)                                                     \
  do {                                                                         \
    CUptiResult err = error;                                                   \
    if (error != CUPTI_SUCCESS) {                                              \
      const char* errstr;                                                      \
      cuptiGetResultString(error, &errstr);                                    \
      fprintf(stderr, "Error: %s %s %d\n", errstr, __FILE__, __LINE__);        \
      if (true)                                                                \
        exit(err);                                                             \
    }                                                                          \
  } while (0)

// TODO: Required for pthread execution, as it needs to return something and
// can't exit with an error. Once callback is added, this can be removed since
// we won't have any threads to worry about.
#define CHECK_CUPTI_ERROR(err)                                                 \
  if (err != CUPTI_SUCCESS) {                                                  \
    const char* errstr;                                                        \
    cuptiGetResultString(err, &errstr);                                        \
    printf(                                                                    \
      "%s:%d:Error %s for CUPTI API function.\n", __FILE__, __LINE__, errstr); \
    return 0;                                                                  \
  }

#define EVENT_NAME "inst_executed"
#define SAMPLE_PERIOD_MS 50
#define ITERATIONS 2000

namespace cuptipp {

#ifdef _WIN32
typedef HANDLE thread_t;
#else
typedef pthread_t thread_t;
#endif

#ifndef CUPTIPP_PROFILER_NAME_SHORT
#define CUPTIPP_PROFILER_NAME_SHORT 64
#endif

typedef CUpti_EventDomainID event_domain_id_t;
typedef CUpti_EventID event_id_t;
typedef CUpti_EventGroup event_group_t;
typedef CUdevice device_t;
typedef CUcontext context_t;

class launch
{
public:
  enum launch_t
  {
    PROFILED,
    NORMAL
  };

  typedef struct
  {
    dim3 blocks_per_grid;
    dim3 threads_per_block;
    size_t dynamic_shared_memory_size;
  } kernel_configs_t;

  template<typename kernel_t, typename... kernel_parameters_t>
  launch(launch_t L,
         const kernel_t& kernel_function,
         kernel_configs_t launch_configs,
         kernel_parameters_t... parameters)
  {
    // Perform a PROFILED cuptipp::launch
    if (L == launch_t::PROFILED) {
      for (int i = 0; i < ITERATIONS; i++) {
        kernel_function<<<launch_configs.blocks_per_grid,
                          launch_configs.threads_per_block,
                          launch_configs.dynamic_shared_memory_size>>>(
          parameters...);
      }
    }

    // Perform a NORMAL cuda kernel launch
    else {
      kernel_function<<<launch_configs.blocks_per_grid,
                        launch_configs.threads_per_block,
                        launch_configs.dynamic_shared_memory_size>>>(
        parameters...);
    }
  }
};

typedef struct
{
  event_group_t event_group;
  event_id_t event_id;
} event_t;

typedef struct
{
  event_t* eventData;
  uint64_t event_value;
} trace_t;

// add a null terminator to the end of a string if the string
// length equals the maximum length (as in that case there was no
// room to write the null terminator)
static void
null_terminator(char* str, size_t len, size_t max_len)
{
  if (len >= max_len) {
    str[max_len - 1] = '\0';
  }
}

std::vector<std::string>
available_events(device_t device)
{
  std::vector<std::string> event_names;

  uint32_t num_domains = 0;
  uint32_t num_events = 0;
  uint32_t total_events = 0;

  size_t size;

  cuptipp::event_domain_id_t* domain_ids;
  cuptipp::event_id_t* event_ids;
  size_t event_size;

  char event_name[CUPTIPP_PROFILER_NAME_SHORT];

  CUPTI_ERROR(cuptiDeviceGetNumEventDomains(device, &num_domains));
  size = sizeof(cuptipp::event_domain_id_t) * num_domains;

  try {
    domain_ids = new cuptipp::event_domain_id_t[size];
  } catch (std::bad_alloc& e) {
    std::cerr << "Bad allocation caught: " << e.what() << '\n';
  }

  CUPTI_ERROR(cuptiDeviceEnumEventDomains(device, &size, domain_ids));

  for (int i = 0; i < num_domains; i++) {
    CUPTI_ERROR(cuptiEventDomainGetNumEvents(domain_ids[i], &num_events));
    total_events += num_events;
  }

  event_size = sizeof(cuptipp::event_id_t) * total_events;

  try {
    event_ids = new cuptipp::event_id_t[event_size];
  } catch (std::bad_alloc& e) {
    std::cerr << "Bad allocation caught: " << e.what() << '\n';
  }

  total_events = 0;
  for (int i = 0; i < num_domains; i++) {
    // Query num of events available in the domain
    CUPTI_ERROR(cuptiEventDomainGetNumEvents(domain_ids[i], &num_events));
    size = num_events * sizeof(cuptipp::event_id_t);
    CUPTI_ERROR(cuptiEventDomainEnumEvents(
      domain_ids[i], &size, event_ids + total_events));
    total_events += num_events;
  }

  for (int i = 0; i < total_events; i++) {
    size = CUPTIPP_PROFILER_NAME_SHORT;
    CUPTI_ERROR(cuptiEventGetAttribute(
      event_ids[i], CUPTI_EVENT_ATTR_NAME, &size, (uint8_t*)event_name));
    null_terminator(event_name, size, CUPTIPP_PROFILER_NAME_SHORT);
    event_names.push_back(event_name);
  }

  delete[] domain_ids;
  delete[] event_ids;

  return std::move(event_names);
}

namespace profile {

// used to signal from the compute thread to the sampling thread
static volatile int testComplete = 0;

static context_t context;
static device_t device;
static const char* eventName;

void*
sample(void* arg)
{
  eventName = EVENT_NAME;
  cuptipp::event_group_t event_group;
  cuptipp::event_id_t event_id;

  size_t bytes_read;
  size_t value_size;
  uint32_t instances = 0;
  uint32_t j = 0;

  uint64_t* event_values = NULL;
  uint64_t event_value = 0;

  uint32_t profile_all = 1;

  CHECK_CUPTI_ERROR(cuptiSetEventCollectionMode(
    context, CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS));

  CHECK_CUPTI_ERROR(cuptiEventGroupCreate(context, &event_group, 0));

  CHECK_CUPTI_ERROR(cuptiEventGetIdFromName(device, eventName, &event_id));

  CHECK_CUPTI_ERROR(cuptiEventGroupAddEvent(event_group, event_id));

  CHECK_CUPTI_ERROR(cuptiEventGroupSetAttribute(
    event_group,
    CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
    sizeof(profile_all),
    &profile_all));

  CHECK_CUPTI_ERROR(cuptiEventGroupEnable(event_group));

  value_size = sizeof(instances);
  CHECK_CUPTI_ERROR(
    cuptiEventGroupGetAttribute(event_group,
                                CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
                                &value_size,
                                &instances));

  bytes_read = sizeof(uint64_t) * instances;

  try {
    event_values = new uint64_t[bytes_read];
  } catch (std::bad_alloc& e) {
    std::cerr << "Bad allocation caught: " << e.what() << '\n';
  }

  while (!testComplete) {
    CHECK_CUPTI_ERROR(cuptiEventGroupReadEvent(event_group,
                                               CUPTI_EVENT_READ_FLAG_NONE,
                                               event_id,
                                               &bytes_read,
                                               event_values));
    if (bytes_read != (sizeof(uint64_t) * instances)) {
      printf("Failed to read value for \"%s\"\n", eventName);
      exit(-1);
    }

    for (j = 0; j < instances; j++) {
      event_value += event_values[j];
    }
    printf("%s: %llu\n", eventName, (unsigned long long)event_value);
#ifdef _WIN32
    Sleep(SAMPLE_PERIOD_MS);
#else
    usleep(SAMPLE_PERIOD_MS * 1000);
#endif
  }

  CHECK_CUPTI_ERROR(cuptiEventGroupDisable(event_group));
  CHECK_CUPTI_ERROR(cuptiEventGroupDestroy(event_group));

  delete[] event_values;

  return NULL;
}

template<typename thread_t>
thread_t*
begin()
{
  testComplete = 0;
  CUDA_ERROR(cudaGetDevice(&device));
  CU_ERROR(cuCtxCreate(&context, 0, device));

#ifdef _WIN32
  static thread_t t;
  t = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)sample, NULL, 0, NULL);
  if (!t) {
    printf("CreateThread failed\n");
    exit(-1);
  }
  return &t;
#else
  static thread_t t;
  int status;
  status = pthread_create(&t, NULL, sample, NULL);
  if (status != 0) {
    perror("pthread_create");
    exit(-1);
  }
  return &t;
#endif
}

template<typename thread_t>
void
end(thread_t* t)
{
  // "signal" the sampling thread to exit and wait for it
  testComplete = 1;
#ifdef _WIN32
  WaitForSingleObject(*t, INFINITE);
#else
  pthread_join(*t, NULL);
#endif

  CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace profiler
} // namespace cuptipp