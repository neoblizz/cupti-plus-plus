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

#define CHECK_CUPTI_ERROR(err)                       \
  if (err != CUPTI_SUCCESS)                                     \
    {                                                           \
      const char *errstr;                                       \
      cuptiGetResultString(err, &errstr);                       \
      printf ("%s:%d:Error %s for CUPTI API function.\n",  \
              __FILE__, __LINE__, errstr);           \
      return 0;                                                 \
    }

#define EVENT_NAME "active_warps"
#define SAMPLE_PERIOD_MS 50
#define ITERATIONS 2000

namespace cuptipp {

#ifndef CUPTIPP_PROFILER_NAME_SHORT
#define CUPTIPP_PROFILER_NAME_SHORT 64
#endif

typedef CUpti_EventDomainID EventDomainID;
typedef CUpti_EventID EventID;
typedef CUpti_EventGroup EventGroup;

enum LaunchT
{
  PROFILED,
  NORMAL
};

typedef struct
{
  dim3 blocks_per_grid;
  dim3 threads_per_block;
  size_t dynamic_shared_memory_size;
} ConfigsT;

template<LaunchT L, typename KernelT, typename... ParametersT>
inline void
launch(const KernelT& kernel_function,
       ConfigsT launch_configs,
       ParametersT... parameters)
{
  // Perform a PROFILED cuptipp::launch
  if (L == LaunchT::PROFILED) {
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

// add a null terminator to the end of a string if the string
// length equals the maximum length (as in that case there was no
// room to write the null terminator)
static void null_terminator(char *str, size_t len, size_t max_len) {
    if (len >= max_len) {
        str[max_len - 1] = '\0';
    }
}

std::vector<std::string>
available_events(CUdevice device)
{
  std::vector<std::string> event_names;

  uint32_t numDomains = 0;
  uint32_t numEvents = 0;
  uint32_t totalEvents = 0;

  size_t size;

  cuptipp::EventDomainID* domainIdArray;
  cuptipp::EventID* eventIdArray;
  size_t event_size;

  char eventName[CUPTIPP_PROFILER_NAME_SHORT];

  CUPTI_ERROR(cuptiDeviceGetNumEventDomains(device, &numDomains));
  size = sizeof(cuptipp::EventDomainID) * numDomains;

  try {
    domainIdArray = new cuptipp::EventDomainID[size];
  } catch (std::bad_alloc& e) {
    std::cerr << "Bad allocation caught: " << e.what() << '\n';
  }

  CUPTI_ERROR(cuptiDeviceEnumEventDomains(device, &size, domainIdArray));

  for (int i = 0; i < numDomains; i++) {
    CUPTI_ERROR(cuptiEventDomainGetNumEvents(domainIdArray[i], &numEvents));
    totalEvents += numEvents;
  }

  event_size = sizeof(cuptipp::EventID) * totalEvents;

  try {
    eventIdArray = new cuptipp::EventID[event_size];
  } catch (std::bad_alloc& e) {
    std::cerr << "Bad allocation caught: " << e.what() << '\n';
  }

  totalEvents = 0;
  for (int i = 0; i < numDomains; i++) {
    // Query num of events available in the domain
    CUPTI_ERROR(cuptiEventDomainGetNumEvents(domainIdArray[i], &numEvents));
    size = numEvents * sizeof(cuptipp::EventID);
    CUPTI_ERROR(cuptiEventDomainEnumEvents(
      domainIdArray[i], &size, eventIdArray + totalEvents));
    totalEvents += numEvents;
  }

  for (int i = 0; i < totalEvents; i++) {
    size = CUPTIPP_PROFILER_NAME_SHORT;
    CUPTI_ERROR(cuptiEventGetAttribute(
      eventIdArray[i], CUPTI_EVENT_ATTR_NAME, &size, (uint8_t*)eventName));
    null_terminator(eventName, size, CUPTIPP_PROFILER_NAME_SHORT);
    event_names.push_back(eventName);
  }

  delete[] domainIdArray;
  delete[] eventIdArray;

  return std::move(event_names);
}

namespace profile {

// used to signal from the compute thread to the sampling thread
static volatile int testComplete = 0;

static CUcontext context;
static CUdevice device;
static const char* eventName;

void *
sample(void* arg)
{
  eventName = EVENT_NAME;
  cuptipp::EventGroup eventGroup;
  cuptipp::EventID eventId;

  size_t bytesRead, valueSize;
  uint32_t numInstances = 0, j = 0;

  uint64_t *eventValues = NULL; 
  uint64_t eventVal = 0;
  
  uint32_t profile_all = 1;

  CHECK_CUPTI_ERROR(cuptiSetEventCollectionMode(
    context, CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS));

  CHECK_CUPTI_ERROR(cuptiEventGroupCreate(context, &eventGroup, 0));

  CHECK_CUPTI_ERROR(cuptiEventGetIdFromName(device, eventName, &eventId));

  CHECK_CUPTI_ERROR(cuptiEventGroupAddEvent(eventGroup, eventId));

  CHECK_CUPTI_ERROR(cuptiEventGroupSetAttribute(
    eventGroup,
    CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
    sizeof(profile_all),
    &profile_all));

  CHECK_CUPTI_ERROR(cuptiEventGroupEnable(eventGroup));

  valueSize = sizeof(numInstances);
  CHECK_CUPTI_ERROR(cuptiEventGroupGetAttribute(eventGroup,
                                          CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
                                          &valueSize,
                                          &numInstances));

  bytesRead = sizeof(uint64_t) * numInstances;

  try {
    eventValues = new uint64_t[bytesRead];
  } catch (std::bad_alloc& e) {
    std::cerr << "Bad allocation caught: " << e.what() << '\n';
  }

  while (!testComplete) {
    CHECK_CUPTI_ERROR(cuptiEventGroupReadEvent(eventGroup,
                                         CUPTI_EVENT_READ_FLAG_NONE,
                                         eventId,
                                         &bytesRead,
                                         eventValues));
    if (bytesRead != (sizeof(uint64_t) * numInstances)) {
      printf("Failed to read value for \"%s\"\n", eventName);
      exit(-1);
    }

    for (j = 0; j < numInstances; j++) {
      eventVal += eventValues[j];
    }
    printf("%s: %llu\n", eventName, (unsigned long long)eventVal);
#ifdef _WIN32
    Sleep(SAMPLE_PERIOD_MS);
#else
    usleep(SAMPLE_PERIOD_MS * 1000);
#endif
  }

  CHECK_CUPTI_ERROR(cuptiEventGroupDisable(eventGroup));
  CHECK_CUPTI_ERROR(cuptiEventGroupDestroy(eventGroup));

  delete[] eventValues;

  return NULL;
}

template<typename ThreadT, typename ContextT, typename DeviceT>
void
begin(ThreadT* t, ContextT ctx, DeviceT dev)
{
  testComplete = 0;
  context = ctx;
  device = dev;

#ifdef _WIN32
  *t = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)sample, NULL, 0, NULL);
  if (!(*t)) {
    printf("CreateThread failed\n");
    exit(-1);
  }
#else
  int status;
  status = pthread_create(t, NULL, sample, NULL);
  if (status != 0) {
    perror("pthread_create");
    exit(-1);
  }
#endif
}

template<typename ThreadT>
void
begin(ThreadT* t)
{
  testComplete = 0;
  CUDA_ERROR(cudaGetDevice(&device));
  CU_ERROR(cuCtxCreate(&context, 0, device));

#ifdef _WIN32
  *t = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)sample, NULL, 0, NULL);
  if (!(*t)) {
    printf("CreateThread failed\n");
    exit(-1);
  }
#else
  int status;
  status = pthread_create(t, NULL, sample, NULL);
  if (status != 0) {
    perror("pthread_create");
    exit(-1);
  }
#endif
}

template<typename ThreadT>
void
end(ThreadT* t)
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