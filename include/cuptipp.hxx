#pragma once

// C/C++
#include <cstddef>
#include <stdio.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

// CUDA Profiling Tools Interface (CUPTI)
#include <cuda_runtime_api.h>
#include <cupti.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <pthread.h>
#endif

#define CHECK_CU_ERROR(err, cufunc)                                            \
  if (err != CUDA_SUCCESS) {                                                   \
    printf("Error %d for CUDA Driver API function '%s'.\n", err, cufunc);      \
    return 0;                                                                  \
  }

#define CHECK_CUPTI_ERROR(err, cuptifunc)                                      \
  if (err != CUPTI_SUCCESS) {                                                  \
    const char* errstr;                                                        \
    cuptiGetResultString(err, &errstr);                                        \
    printf("%s:%d:Error %s for CUPTI API function '%s'.\n",                    \
           __FILE__,                                                           \
           __LINE__,                                                           \
           errstr,                                                             \
           cuptifunc);                                                         \
    return 0;                                                                  \
  }

#define EVENT_NAME "l2_subp0_read_sector_misses"
#define SAMPLE_PERIOD_MS 50
#define ITERATIONS 2000

namespace cuptipp {

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

namespace profile {

// used to signal from the compute thread to the sampling thread
static volatile int testComplete = 0;

static CUcontext context;
static CUdevice device;
static const char* eventName;

template<typename T>
T*
sample(void* arg)
{
  eventName = EVENT_NAME;
  CUptiResult cuptiErr;
  CUpti_EventGroup eventGroup;
  CUpti_EventID eventId;
  size_t bytesRead, valueSize;
  uint32_t numInstances = 0, j = 0;
  uint64_t *eventValues = NULL, eventVal = 0;
  uint32_t profile_all = 1;

  cuptiErr = cuptiSetEventCollectionMode(
    context, CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiSetEventCollectionMode");

  cuptiErr = cuptiEventGroupCreate(context, &eventGroup, 0);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupCreate");

  cuptiErr = cuptiEventGetIdFromName(device, eventName, &eventId);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGetIdFromName");

  cuptiErr = cuptiEventGroupAddEvent(eventGroup, eventId);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupAddEvent");

  cuptiErr = cuptiEventGroupSetAttribute(
    eventGroup,
    CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
    sizeof(profile_all),
    &profile_all);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupSetAttribute");

  cuptiErr = cuptiEventGroupEnable(eventGroup);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupEnable");

  valueSize = sizeof(numInstances);
  cuptiErr = cuptiEventGroupGetAttribute(eventGroup,
                                         CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
                                         &valueSize,
                                         &numInstances);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupGetAttribute");

  bytesRead = sizeof(uint64_t) * numInstances;
  eventValues = (uint64_t*)malloc(bytesRead);
  if (eventValues == NULL) {
    printf("%s:%d: Failed to allocate memory.\n", __FILE__, __LINE__);
    exit(-1);
  }

  while (!testComplete) {
    cuptiErr = cuptiEventGroupReadEvent(
      eventGroup, CUPTI_EVENT_READ_FLAG_NONE, eventId, &bytesRead, eventValues);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupReadEvent");
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

  cuptiErr = cuptiEventGroupDisable(eventGroup);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupDisable");

  cuptiErr = cuptiEventGroupDestroy(eventGroup);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupDestroy");

  free(eventValues);
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
  cudaGetDevice(&device);
  cuCtxCreate(&context, 0, device);

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

  cudaDeviceSynchronize();
}

} // namespace profiler
} // namespace cuptipp