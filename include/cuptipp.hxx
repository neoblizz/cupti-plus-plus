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
  if (error != CUDA_SUCCESS) {                                                 \
    const char* errstr;                                                        \
    cuGetErrorString(error, &errstr);                                          \
    fprintf(stderr, "Error: %s %s %d\n", errstr, __FILE__, __LINE__);          \
    if (true)                                                                  \
      exit(error);                                                             \
  }

#define CUDA_ERROR(error)                                                      \
  if (error != cudaSuccess) {                                                  \
    const char* errstr = cudaGetErrorString(error);                            \
    fprintf(stderr, "Error: %s %s %d\n", errstr, __FILE__, __LINE__);          \
    if (true)                                                                  \
      exit(error);                                                             \
  }

#define CUPTI_ERROR(error)                                                     \
  if (error != CUPTI_SUCCESS) {                                                \
    const char* errstr;                                                        \
    cuptiGetResultString(error, &errstr);                                      \
    fprintf(stderr, "Error: %s %s %d\n", errstr, __FILE__, __LINE__);          \
    if (true)                                                                  \
      exit(error);                                                             \
  }

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

#define NAME_SHORT 64
#define NAME_LONG 128

#define DESC_SHORT 512
#define DESC_LONG 2048
#define CATEGORY_LENGTH sizeof(CUpti_EventCategory)

typedef CUpti_EventDomainID event_domain_id_t;
typedef CUpti_EventID event_id_t;
typedef CUpti_EventGroup event_group_t;
typedef std::string event_name_t;
typedef std::string event_short_desc_t;
typedef std::string event_long_desc_t;
typedef CUpti_EventCategory event_category_t;

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
  event_name_t event_name;
  event_short_desc_t event_short_desc;
  event_long_desc_t event_long_desc;
  event_category_t event_category;
} event_t;

typedef struct
{
  event_t event_data;
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

std::vector<trace_t>
available_events(device_t device)
{
  std::vector<trace_t> traces;

  uint32_t num_domains = 0;
  uint32_t num_events = 0;
  uint32_t total_events = 0;

  size_t size;

  cuptipp::event_domain_id_t* domain_ids;
  cuptipp::event_id_t* event_ids;
  size_t event_size;

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

  // Fetch and populate the trace information
  trace_t new_trace;
  char event_name[NAME_SHORT];
  char event_short_d[DESC_SHORT];
  char event_long_d[DESC_LONG];

  for (int i = 0; i < total_events; i++) {
    size = NAME_SHORT;
    CUPTI_ERROR(cuptiEventGetAttribute(
      event_ids[i], CUPTI_EVENT_ATTR_NAME, &size, (uint8_t*)event_name));
    null_terminator(event_name, size, NAME_SHORT);

    size = DESC_SHORT;
    CUPTI_ERROR(cuptiEventGetAttribute(event_ids[i],
                                       CUPTI_EVENT_ATTR_SHORT_DESCRIPTION,
                                       &size,
                                       (uint8_t*)event_short_d));
    null_terminator(event_short_d, size, DESC_SHORT);

    size = DESC_LONG;
    CUPTI_ERROR(cuptiEventGetAttribute(event_ids[i],
                                       CUPTI_EVENT_ATTR_LONG_DESCRIPTION,
                                       &size,
                                       (uint8_t*)event_long_d));
    null_terminator(event_long_d, size, DESC_LONG);

    size = CATEGORY_LENGTH;
    CUPTI_ERROR(cuptiEventGetAttribute(event_ids[i],
                                       CUPTI_EVENT_ATTR_CATEGORY,
                                       &size,
                                       (&new_trace.event_data.event_category)));

    new_trace.event_data.event_id = event_ids[i];
    new_trace.event_data.event_name = event_name;
    new_trace.event_data.event_short_desc = event_short_d;
    new_trace.event_data.event_long_desc = event_long_d;
    traces.push_back(new_trace);
  }

  {
    for (const auto e : traces) {
      std::cout << e.event_data.event_id << ' ';
      std::cout << e.event_data.event_name << '\n';
      std::cout << e.event_data.event_short_desc << '\n';
      std::cout << e.event_data.event_long_desc << '\n';
      std::cout << e.event_data.event_category << '\n';
    }
  }

  delete[] domain_ids;
  delete[] event_ids;

  return std::move(traces);
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