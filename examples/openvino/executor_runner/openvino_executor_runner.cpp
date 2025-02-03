#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>

#include <gflags/gflags.h>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

#include <OpenvinoBackend.hpp>  // Include the OpenVINO backend header

// Define a fixed-size memory pool for the method allocator (4 MB)
static uint8_t method_allocator_pool[4 * 1024U * 1024U]; // 4 MB

// Define command-line flags for model path, the number of iterations, input list path, and output folder path
DEFINE_string(
    model_path,
    "",
    "Path to the model serialized in flatbuffer format (required).");
DEFINE_int32(
    num_iter,
    1,
    "Number of inference iterations (default is 1).");
DEFINE_string(
    input_list_path,
    "",
    "Path to the input list file which includes the list of raw input tensor files (optional).");
DEFINE_string(
    output_folder_path,
    "",
    "Path to the output folder to save raw output tensor files (optional).");

using executorch::extension::FileDataLoader;
using executorch::extension::prepare_input_tensors;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::TensorInfo;

int main(int argc, char** argv) {
  // Initialize the runtime environment
  executorch::runtime::runtime_init();

  // Initialize OpenVINO backend directly
  executorch::backends::openvino::OpenvinoBackend backend;
  executorch::runtime::Backend backend_id{"OpenvinoBackend", &backend};
  Error registered = executorch::runtime::register_backend(backend_id);

  // Parse command-line arguments and flags
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Check if the model path is provided
  if (FLAGS_model_path.empty()) {
    std::cerr << "Error: --model_path is required." << std::endl;
    std::cerr << "Usage: " << argv[0]
              << " --model_path=<path_to_model> --num_iter=<iterations>" << std::endl;
    return 1;
  }

  // Retrieve the model path and number of iterations
  const char* model_path = FLAGS_model_path.c_str();
  int num_iterations = FLAGS_num_iter;
  std::cout << "Model path: " << model_path << std::endl;
  std::cout << "Number of iterations: " << num_iterations << std::endl;

  // Load the model using FileDataLoader
  Result<FileDataLoader> loader = FileDataLoader::from(model_path);
  ET_CHECK_MSG(
      loader.ok(),
      "FileDataLoader::from() failed: 0x%" PRIx32,
      static_cast<uint32_t>(loader.error()));

  // Load the program from the loaded model
  Result<Program> program = Program::load(&loader.get());
  if (!program.ok()) {
    ET_LOG(Error, "Failed to parse model file %s", model_path);
    return 1;
  }
  ET_LOG(Info, "Model file %s is loaded.", model_path);

  // Retrieve the method name from the program (assumes the first method is used)
  const char* method_name = nullptr;
  {
    const auto method_name_result = program->get_method_name(0);
    ET_CHECK_MSG(method_name_result.ok(), "Program has no methods");
    method_name = *method_name_result;
  }
  ET_LOG(Info, "Using method %s", method_name);

  // Retrieve metadata about the method
  Result<MethodMeta> method_meta = program->method_meta(method_name);
  ET_CHECK_MSG(
      method_meta.ok(),
      "Failed to get method_meta for %s: 0x%" PRIx32,
      method_name,
      static_cast<uint32_t>(method_meta.error()));

  // Set up a memory allocator for the method
  MemoryAllocator method_allocator{
      MemoryAllocator(sizeof(method_allocator_pool), method_allocator_pool)};

  // Prepare planned buffers for memory planning
  std::vector<std::unique_ptr<uint8_t[]>> planned_buffers;
  std::vector<Span<uint8_t>> planned_spans;
  size_t num_memory_planned_buffers = method_meta->num_memory_planned_buffers();
  for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
    size_t buffer_size =
        static_cast<size_t>(method_meta->memory_planned_buffer_size(id).get());
    ET_LOG(Info, "Setting up planned buffer %zu, size %zu.", id, buffer_size);
    planned_buffers.push_back(std::make_unique<uint8_t[]>(buffer_size));
    planned_spans.push_back({planned_buffers.back().get(), buffer_size});
  }
  HierarchicalAllocator planned_memory(
      {planned_spans.data(), planned_spans.size()});

  // Set up a memory manager using the method allocator and planned memory
  MemoryManager memory_manager(&method_allocator, &planned_memory);

  // Load the method into the program
  Result<Method> method = program->load_method(method_name, &memory_manager);
  ET_CHECK_MSG(
      method.ok(),
      "Loading of method %s failed with status 0x%" PRIx32,
      method_name,
      static_cast<uint32_t>(method.error()));
  ET_LOG(Info, "Method loaded.");

  // Prepare the input tensors for the method
  auto inputs = prepare_input_tensors(*method);
  ET_CHECK_MSG(
      inputs.ok(),
      "Could not prepare inputs: 0x%" PRIx32,
      static_cast<uint32_t>(inputs.error()));

  // If the input path list is provided, read input tensors from the files
  if (!(FLAGS_input_list_path.empty())) {
    const char* input_list_path = FLAGS_input_list_path.c_str();
    ET_LOG(Info, "Loading input tensors from the list provided in %s.", input_list_path);
    Error status = Error::Ok;
    std::vector<EValue> inputs(method->inputs_size());
    ET_LOG(Info, "%zu inputs: ", inputs.size());
    status = method->get_inputs(inputs.data(), inputs.size());
    ET_CHECK(status == Error::Ok);

    auto split = [](std::string s, std::string delimiter) {
      size_t pos_start = 0, pos_end, delim_len = delimiter.length();
      std::string token;
      std::vector<std::string> res;

      while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back(token);
      }
      res.push_back(s.substr(pos_start));
      return res;
    };

    // Read raw input tensor file names from input list file and
    // iterate each raw input tensor file to read values
    std::ifstream input_list(input_list_path);
    if (input_list.is_open()) {
      size_t num_inputs = method->inputs_size();
      std::string file_path;
      while (std::getline(input_list, file_path)) {
        auto input_files = split(file_path, " ");
        if (input_files.size() == 0) {
          break;
        }
        for (int input_index = 0; input_index < num_inputs; ++input_index) {
            MethodMeta method_meta = method->method_meta();
            Result<TensorInfo> tensor_meta =
                method_meta.input_tensor_meta(input_index);
            auto input_data_ptr = inputs[input_index].toTensor().data_ptr<char>();

            std::ifstream fin(input_files[input_index], std::ios::binary);
            fin.seekg(0, fin.end);
            size_t file_size = fin.tellg();

            ET_CHECK_MSG(
                file_size == tensor_meta->nbytes(),
                "Input(%d) size mismatch. file bytes: %zu, tensor bytes: %zu",
                input_index,
                file_size,
                tensor_meta->nbytes());

            fin.seekg(0, fin.beg);
            fin.read(
                static_cast<char*>(input_data_ptr),
                file_size);
            fin.close();
        }
      }
    } else {
      ET_CHECK_MSG(false,
          "Failed to read input list file: %s",
          input_list_path);
    }
  }
  ET_LOG(Info, "Inputs prepared.");

  // Measure execution time for inference
  auto before_exec = std::chrono::high_resolution_clock::now();
  Error status = Error::Ok;
  for (int i = 0; i < num_iterations; ++i) {
    status = method->execute();
  }
  auto after_exec = std::chrono::high_resolution_clock::now();
  double elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(
                            after_exec - before_exec)
                            .count() / 1000.0;

  // Log execution time and average time per iteration
  ET_LOG(
      Info,
      "%d inference took %f ms, avg %f ms",
      num_iterations,
      elapsed_time,
      elapsed_time / static_cast<float>(num_iterations));
  ET_CHECK_MSG(
      status == Error::Ok,
      "Execution of method %s failed with status 0x%" PRIx32,
      method_name,
      static_cast<uint32_t>(status));
  ET_LOG(Info, "Model executed successfully.");

  // Retrieve and print the method outputs
  std::vector<EValue> outputs(method->outputs_size());
  ET_LOG(Info, "%zu Number of outputs: ", outputs.size());
  status = method->get_outputs(outputs.data(), outputs.size());
  ET_CHECK(status == Error::Ok);

  // If output folder path is provided, save output tensors
  // into raw tensor files.
  if (!(FLAGS_output_folder_path.empty())) {
    const char* output_folder_path = FLAGS_output_folder_path.c_str();
    ET_LOG(Info, "Saving output tensors into the output folder: %s.", output_folder_path);
    for (size_t output_index = 0; output_index < method->outputs_size();
         output_index++) {
      auto output_tensor = outputs[output_index].toTensor();
      auto output_file_name = std::string(output_folder_path) + "/output_" +
          std::to_string(output_index) + ".raw";
      std::ofstream fout(output_file_name.c_str(), std::ios::binary);
      fout.write(
          output_tensor.const_data_ptr<char>(), output_tensor.nbytes());
      fout.close();
    }
  }

  return 0;
}
