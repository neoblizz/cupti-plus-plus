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

static const char* dummy_kernel_name = "^^ DUMMY ^^";

typedef CUpti_EventDomainID event_domain_id_t;
typedef CUpti_EventID event_id_t;
typedef CUpti_EventGroup event_group_t;
typedef std::string event_name_t;
typedef std::string event_short_desc_t;
typedef std::string event_long_desc_t;
typedef CUpti_EventCategory event_category_t;
typedef uint64_t event_value_t;
typedef CUpti_EventGroupSets event_group_sets_t;
typedef CUpti_EventGroupSet event_group_set_t;

typedef CUpti_SubscriberHandle subscriber_t;
typedef CUpti_CallbackDomain call_back_domain_t;
typedef CUpti_CallbackId call_back_id_t;
typedef CUpti_CallbackData call_back_data_t;
typedef CUpti_CallbackFunc call_back_function_t;

typedef CUdevice device_t;
typedef CUcontext context_t;
typedef int iteration_t;

typedef struct
{
  dim3 blocks_per_grid;
  dim3 threads_per_block;
  size_t dynamic_shared_memory_size;
  cudaStream_t stream;
} kernel_configs_t;

enum launch_t
{
  PROFILED,
  NORMAL
};

typedef struct
{
  event_group_t event_group;
  event_id_t event_id;
  event_name_t event_name;
  event_short_desc_t event_short_desc;
  event_long_desc_t event_long_desc;
  event_category_t event_category;
  event_value_t event_value;
} event_t;

typedef struct
{
  event_group_set_t* event_groups;
  uint32_t num_events;
  std::vector<event_id_t> event_ids;
  std::vector<event_value_t> event_values;
} pass_t;

struct trace_t
{
  std::vector<pass_t> pass_data;
  std::string kernel_name;

  trace_t()
    : current_pass(0)
  {}

  int event_passes;
  int current_pass;
  int total_passes;
  device_t device;

  std::vector<event_value_t> event_values;
};

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

std::vector<event_name_t>
available_events(device_t device)
{
  std::vector<event_t> events;

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
  event_t new_event;
  char event_name[NAME_SHORT];
  char event_short_d[DESC_SHORT];
  char event_long_d[DESC_LONG];
  event_category_t event_category;
  std::vector<event_name_t> event_names;

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
    CUPTI_ERROR(cuptiEventGetAttribute(
      event_ids[i], CUPTI_EVENT_ATTR_CATEGORY, &size, &event_category));

    new_event.event_id = event_ids[i];
    new_event.event_name = event_name;
    new_event.event_category = event_category;
    new_event.event_short_desc = event_short_d;
    new_event.event_long_desc = event_long_d;
    events.push_back(new_event);

    // XXX: For now, return just the event names.
    event_names.push_back(event_name);
  }

#if DEBUG
  {
    for (const auto e : events) {
      std::cout << "Event ID: " << e.event_id << " ";
      std::cout << "Category: " << e.event_category << "\n";
      std::cout << "Event Name: " << e.event_name << "\n";
      std::cout << e.event_short_desc << "\n";
      std::cout << e.event_long_desc << "\n\n";
    }
  }
#endif

  delete[] domain_ids;
  delete[] event_ids;

  return std::move(event_names);
}

void CUPTIAPI
get_event_value(void* user_data,
                call_back_domain_t cb_domain,
                call_back_id_t cb_id,
                const call_back_data_t* cb_info)
{

  // This callback is enabled only for launch so we shouldn't see anything
  if ((cb_id != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) &&
      (cb_id != CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000)) {
    printf("%s:%d: unexpected cbid %d\n", __FILE__, __LINE__, cb_id);
    exit(-1);
  }

  CUDA_ERROR(cudaDeviceSynchronize());

  const char* current_kernel_name = cb_info->symbolName;

  // XXX: Skip execution if kernel name is NULL string
  if (!current_kernel_name) {
    std::cout << "Kernel symbol is a NULL string.";
    return;
  }

  // else {
  //   std::cout << "Kernel Symbol: " << current_kernel_name << "\n";
  // }

  std::map<std::string, trace_t>* kernel_data =
    (std::map<std::string, trace_t>*)user_data;

  if (cb_info->callbackSite == CUPTI_API_ENTER) {
    CUDA_ERROR(cudaDeviceSynchronize());
    if (kernel_data->count(current_kernel_name) == 0) {
      trace_t temp_trace = (*kernel_data)[dummy_kernel_name];
      trace_t _trace = temp_trace;

      _trace.kernel_name = current_kernel_name;
      auto& pass_data = _trace.pass_data;

      CUPTI_ERROR(cuptiSetEventCollectionMode(
        cb_info->context, CUPTI_EVENT_COLLECTION_MODE_KERNEL));

      for (int i = 0; i < pass_data[0].event_groups->numEventGroups; i++) {
        uint32_t profile_all = 1;
        CUPTI_ERROR(cuptiEventGroupSetAttribute(
          pass_data[0].event_groups->eventGroups[i],
          CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
          sizeof(profile_all),
          &profile_all));
        CUPTI_ERROR(
          cuptiEventGroupEnable(pass_data[0].event_groups->eventGroups[i]));

        (*kernel_data)[current_kernel_name] = _trace;
      }
    } else {
      auto& current_kernel = (*kernel_data)[current_kernel_name];
      auto const& pass_data = current_kernel.pass_data;

      int current_pass = current_kernel.current_pass;
      if (current_pass >= current_kernel.total_passes)
        return;

      CUPTI_ERROR(cuptiSetEventCollectionMode(
        cb_info->context, CUPTI_EVENT_COLLECTION_MODE_KERNEL));

      for (int i = 0; i < pass_data[current_pass].event_groups->numEventGroups;
           i++) {
        uint32_t profile_all = 1;
        CUPTI_ERROR(cuptiEventGroupSetAttribute(
          pass_data[current_pass].event_groups->eventGroups[i],
          CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
          sizeof(profile_all),
          &profile_all));
        CUPTI_ERROR(cuptiEventGroupEnable(
          pass_data[current_pass].event_groups->eventGroups[i]));
      }
    }
  } else if (cb_info->callbackSite == CUPTI_API_EXIT) {
    auto& current_kernel = (*kernel_data)[current_kernel_name];
    int current_pass = current_kernel.current_pass;

    if (current_pass >= current_kernel.total_passes)
      return;

    auto& pass_data = current_kernel.pass_data[current_pass];

    for (int i = 0; i < pass_data.event_groups->numEventGroups; i++) {
      event_group_t group = pass_data.event_groups->eventGroups[i];
      event_domain_id_t group_domain;
      uint32_t num_events, num_instances, num_total_instances;
      size_t group_domain_size = sizeof(group_domain);
      size_t num_events_size = sizeof(num_events);
      size_t num_instances_size = sizeof(num_instances);
      size_t num_total_instances_size = sizeof(num_total_instances);

      CUPTI_ERROR(
        cuptiEventGroupGetAttribute(group,
                                    CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID,
                                    &group_domain_size,
                                    &group_domain));
      CUPTI_ERROR(cuptiDeviceGetEventDomainAttribute(
        current_kernel.device,
        group_domain,
        CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT,
        &num_total_instances_size,
        &num_total_instances));
      CUPTI_ERROR(
        cuptiEventGroupGetAttribute(group,
                                    CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
                                    &num_instances_size,
                                    &num_instances));
      CUPTI_ERROR(cuptiEventGroupGetAttribute(group,
                                              CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
                                              &num_events_size,
                                              &num_events));

      size_t event_ids_size = num_events * sizeof(event_id_t);
      event_id_t* event_ids = new event_id_t[event_ids_size];
      CUPTI_ERROR(cuptiEventGroupGetAttribute(
        group, CUPTI_EVENT_GROUP_ATTR_EVENTS, &event_ids_size, event_ids));

      size_t value_size = sizeof(event_value_t) * num_instances;
      event_value_t* values = new event_value_t[value_size];

      for (int j = 0; j < num_events; j++) {
        CUPTI_ERROR(cuptiEventGroupReadEvent(group,
                                             CUPTI_EVENT_READ_FLAG_NONE,
                                             event_ids[j],
                                             &value_size,
                                             values));

        // sum collect event values from all instances
        event_value_t sum = 0;
        for (int k = 0; k < num_instances; k++)
          sum += values[k];

        // normalize the event value to represent the total number of
        // domain instances on the device
        event_value_t normalized = (sum * num_total_instances) / num_instances;

        pass_data.event_ids.push_back(event_ids[j]);
        pass_data.event_values.push_back(normalized);

#if DEBUG
        // debug: print event values
        {
          char event_name[NAME_LONG];
          size_t event_name_size = NAME_LONG;
          CUPTI_ERROR(cuptiEventGetAttribute(
            event_ids[j], CUPTI_EVENT_ATTR_NAME, &event_name_size, event_name));
          cuptipp::null_terminator(event_name, event_name_size, NAME_SHORT);

          // if (num_instances > 1) {
          //   for (int k = 0; k < num_instances; k++) {
          //     std::cout << values[k] << ", ";
          //   }
          // }
          // std::cout << "\n";
          std::cout << "\t" << event_name << "\n\t\t = (" << sum << " * "
                    << num_total_instances << ") / " << num_instances
                    << "\t= " << normalized << "\n";
        }
#endif
      }

      delete[] values;
      delete[] event_ids;
    }

    for (int i = 0; i < pass_data.event_groups->numEventGroups; i++) {
      CUPTI_ERROR(
        cuptiEventGroupDisable(pass_data.event_groups->eventGroups[i]));
    }
    ++(*kernel_data)[current_kernel_name].current_pass;
  }
}

struct profile
{

  using event_t = cuptipp::event_t;
  using pass_t = cuptipp::pass_t;
  using event_name_t = cuptipp::event_name_t;
  using event_group_sets_t = cuptipp::event_group_sets_t;

  using trace_t = cuptipp::trace_t;
  using device_t = cuptipp::device_t;
  using subscriber_t = cuptipp::subscriber_t;

  using launch_t = cuptipp::launch_t;
  using iteration_t = cuptipp::iteration_t;

  profile(const std::vector<event_name_t>& event_names)
    : p_event_names(event_names)
    , p_num_events(event_names.size())
    , p_event_passes(0)
  {
    CUDA_ERROR(cudaGetDevice(&p_device));
    CU_ERROR(cuCtxCreate(&p_context, 0, p_device));

    CUPTI_ERROR(
      cuptiSubscribe(&p_subscriber,
                     (cuptipp::call_back_function_t)cuptipp::get_event_value,
                     &p_trace));

    CUPTI_ERROR(cuptiEnableCallback(1,
                                    p_subscriber,
                                    CUPTI_CB_DOMAIN_RUNTIME_API,
                                    CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
    CUPTI_ERROR(
      cuptiEnableCallback(1,
                          p_subscriber,
                          CUPTI_CB_DOMAIN_RUNTIME_API,
                          CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));

    // Creating event group for profiling
    event_id_t* event_ids = new event_id_t[p_num_events];
    // event_id_t *event_ids = (event_id_t*)calloc(p_num_events,
    // sizeof(event_id_t));
    for (int i = 0; i < p_num_events; i++) {
      CUPTI_ERROR(cuptiEventGetIdFromName(
        p_device, p_event_names[i].c_str(), &event_ids[i]));
    }

    p_event_ids.resize(p_num_events);

    if (p_num_events > 0) {
      CUPTI_ERROR(cuptiEventGroupSetsCreate(p_context,
                                            sizeof(event_id_t) * p_num_events,
                                            event_ids,
                                            &p_event_pass_data));
      p_event_passes = p_event_pass_data->numSets;
      std::copy(event_ids, event_ids + p_num_events, p_event_ids.begin());
    }

    trace_t temp_trace;
    temp_trace.kernel_name = dummy_kernel_name;
    temp_trace.event_passes = p_event_passes;
    temp_trace.device = p_device;
    temp_trace.total_passes = p_event_passes;
    temp_trace.pass_data.resize(p_event_passes);

    auto& pass_data = temp_trace.pass_data;
    for (int i = 0; i < p_event_passes; i++) {
      int total_events = 0;
      uint32_t num_events = 0;
      size_t num_events_size = sizeof(num_events);
      for (int j = 0; j < p_event_pass_data->sets[i].numEventGroups; j++) {
        CUPTI_ERROR(
          cuptiEventGroupGetAttribute(p_event_pass_data->sets[i].eventGroups[j],
                                      CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
                                      &num_events_size,
                                      &num_events));
        total_events += num_events;
      }
      pass_data[i].event_groups = p_event_pass_data->sets + i;
      pass_data[i].num_events = total_events;
    }

    p_trace[dummy_kernel_name] = temp_trace;
    delete[] event_ids;
  }

  ~profile() {}

  template<typename kernel_t, typename... kernel_parameters_t>
  void launch(const iteration_t NUM_RUNS,
              launch_t L,
              const kernel_t& kernel_function,
              kernel_configs_t launch_configs,
              kernel_parameters_t... parameters)
  {
    // Perform a PROFILED cuptipp::launch
    if (L == launch_t::PROFILED) {
      for (int i = 0; i < NUM_RUNS; i++) {
        kernel_function<<<launch_configs.blocks_per_grid,
                          launch_configs.threads_per_block,
                          launch_configs.dynamic_shared_memory_size,
                          launch.configs.stream>>>(
          parameters...);
      }
    }

    // Perform a NORMAL cuda kernel launch
    else {
      kernel_function<<<launch_configs.blocks_per_grid,
                        launch_configs.threads_per_block,
                        launch_configs.dynamic_shared_memory_size,
                        launch.configs.stream>>>(
        parameters...);
    }
  }

  // XXX: Implement a display stats feature
  void display() {}

  // XXX: Not sure what this can be used for right now...
  void begin() {}

  // XXX: Subscriber, callback clean-up.
  void end()
  {
    // Disable callback and unsubscribe
    CUPTI_ERROR(cuptiEnableCallback(0,
                                    p_subscriber,
                                    CUPTI_CB_DOMAIN_RUNTIME_API,
                                    CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
    CUPTI_ERROR(
      cuptiEnableCallback(0,
                          p_subscriber,
                          CUPTI_CB_DOMAIN_RUNTIME_API,
                          CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));
    CUPTI_ERROR(cuptiUnsubscribe(p_subscriber));
  }

  iteration_t get_iterations() { return p_event_passes; }

  std::vector<std::string> get_kernel_names()
  {
    if (p_kernel_names.size() == 0) {
      for (auto const& k : p_trace) {
        if (k.first == dummy_kernel_name)
          continue;
        p_kernel_names.push_back(k.first);
      }
    }
    return p_kernel_names;
  }

  std::vector<event_value_t> get_event_values(const char* kernel_name)
  {
    if (p_num_events <= 0)
      return std::vector<event_value_t>{};

    // Fetch the data from different passes
    auto& trace = p_trace[kernel_name];
    auto& data = trace.pass_data;
    auto num_passes = trace.total_passes;

    int total_events = 0;
    for (int i = 0; i < num_passes; i++) {
      total_events = data[i].num_events;
    }

    event_id_t* event_ids = new event_id_t[total_events];
    event_value_t* event_values = new event_value_t[total_events];

    int offset = 0;
    for (int i = 0; i < num_passes; i++) {
      std::copy(
        data[i].event_ids.begin(), data[i].event_ids.end(), event_ids + offset);
      std::copy(data[i].event_values.begin(),
                data[i].event_values.end(),
                event_values + offset);
      offset += data[i].num_events;
    }

    for (int i = 0; i < p_num_events; i++) {
      trace.event_values.push_back(event_values[i]);
    }

    return trace.event_values;
  }

private:
  event_t p_events;
  int p_num_events;
  const std::vector<event_name_t>& p_event_names;
  std::vector<event_id_t> p_event_ids;
  event_group_sets_t* p_event_pass_data;

  context_t p_context;
  device_t p_device;
  subscriber_t p_subscriber;

  iteration_t p_event_passes;

  std::map<std::string, trace_t> p_trace;
  std::vector<std::string> p_kernel_names;
  int p_num_kernels;

}; // struct profile

} // namespace cuptipp