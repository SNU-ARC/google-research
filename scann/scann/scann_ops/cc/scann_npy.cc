// Copyright 2021 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "scann/scann_ops/cc/scann_npy.h"
#include <tuple> // ADD FOR SOW INFO

namespace research_scann {

template <typename T>
inline pybind11::array_t<T> VectorToNumpy2D(const std::vector<T>& v,
                                            size_t dim1) {
  size_t dim2 = v.size() / dim1;
  return pybind11::array_t<T>({dim1, dim2}, v.data());
}

void RuntimeErrorIfNotOk(const char* prefix, const Status& status) {
  if (!status.ok()) {
    std::string msg = prefix + std::string(status.error_message());
    throw std::runtime_error(msg);
  }
}

template <typename T>
ConstSpan<T> NumpyToSpan(const np_row_major_arr<T> numpy, size_t expected_dim,
                         const string_view name) {
  if (numpy.ndim() != expected_dim)
    throw std::invalid_argument(absl::StrFormat(
        "%s should be %d-dimensional, but got %d-dimensional input", name,
        expected_dim, numpy.ndim()));
  return ConstSpan<T>(numpy.data(), numpy.size());
}

ScannNumpy::ScannNumpy(
    std::optional<const np_row_major_arr<float>> np_dataset,
    std::optional<const np_row_major_arr<int32_t>> datapoint_to_token,
    std::optional<const np_row_major_arr<uint8_t>> hashed_dataset,
    std::optional<const np_row_major_arr<int8_t>> int8_dataset,
    std::optional<const np_row_major_arr<float>> int8_multipliers,
    std::optional<const np_row_major_arr<float>> dp_norms,
    const std::string& artifacts_dir,
    const std::string& coarse_path,
    const std::string& fine_path) {
  DatapointIndex n_points = kInvalidDatapointIndex;
  ConstSpan<float> dataset;
  if (np_dataset) {
    dataset = NumpyToSpan(*np_dataset, 2, "Dataset");
    n_points = np_dataset->shape()[0];
  }

  ConstSpan<int32_t> tokenization;
  if (datapoint_to_token) {
    tokenization =
        NumpyToSpan(*datapoint_to_token, 1, "Datapoint tokenization");
    n_points = datapoint_to_token->shape()[0];
  }

  ConstSpan<uint8_t> hashed_span;
  if (hashed_dataset) {
    hashed_span = NumpyToSpan(*hashed_dataset, 2, "Hashed dataset");
    n_points = hashed_dataset->shape()[0];
  }

  ConstSpan<int8_t> int8_span;
  ConstSpan<float> mult_span, norm_span;
  if (int8_dataset) {
    int8_span = NumpyToSpan(*int8_dataset, 2, "Int8-quantized dataset");
    n_points = int8_dataset->shape()[0];
  }
  if (int8_multipliers)
    mult_span =
        NumpyToSpan(*int8_multipliers, 1, "Int8 quantization multipliers");
  if (dp_norms)
    norm_span = NumpyToSpan(*dp_norms, 1, "Datapoint squared L2 norms");

  RuntimeErrorIfNotOk(
      "Error initializing searcher: ",
      scann_.Initialize(dataset, tokenization, hashed_span, int8_span,
                        mult_span, norm_span, n_points, artifacts_dir, coarse_path, fine_path));
}

ScannNumpy::ScannNumpy(const np_row_major_arr<float>& np_dataset,
                       const np_row_major_arr<float>& train_set,
                       const std::string& config, const bool& load_coarse, const std::string& coarse_path, const bool& load_fine, const std::string& fine_path, int training_threads) {
  if (np_dataset.ndim() != 2)
    throw std::invalid_argument("Dataset input must be two-dimensional");
  ConstSpan<float> dataset(np_dataset.data(), np_dataset.size());
  RuntimeErrorIfNotOk("Error initializing searcher: ",
                      scann_.Initialize(dataset, train_set, np_dataset.shape()[0], train_set.shape()[0], config, load_coarse, coarse_path, load_fine, fine_path,
                                        training_threads));
}

std::tuple<pybind11::array_t<float>, pybind11::array_t<DatapointIndex>, pybind11::array_t<float>>
ScannNumpy::Search(const np_row_major_arr<float>& query, int final_nn,
                   int pre_reorder_nn, int leaves) {
  if (query.ndim() != 1)
    throw std::invalid_argument("Query must be one-dimensional");

  pybind11::array_t<unsigned long long int> SOW({static_cast<long>(1 * (leaves + 1)), 1L});  // ADD FOR SOW INFO, one query at a time
  auto sow_ptr = reinterpret_cast<unsigned long long int*>(SOW.request().ptr);
  size_t begin = 0;
  size_t curSize = 0;

  DatapointPtr<float> ptr(nullptr, query.data(), query.size(), query.size());
  NNResultsVector res;
  auto status = scann_.Search(ptr, &res, final_nn, pre_reorder_nn, leaves, sow_ptr, begin, curSize);
  RuntimeErrorIfNotOk("Error during search: ", status);

  // pybind11::array_t<DatapointIndex> indices(res.size());
  // pybind11::array_t<float> distances(res.size());
  pybind11::array_t<DatapointIndex> indices(final_nn);
  pybind11::array_t<float> distances(final_nn);
  auto idx_ptr = reinterpret_cast<DatapointIndex*>(indices.request().ptr);
  auto dis_ptr = reinterpret_cast<float*>(distances.request().ptr);
  scann_.ReshapeNNResult(res, idx_ptr, dis_ptr, final_nn);
  return {SOW, indices, distances};
}

std::tuple<pybind11::array_t<float>, pybind11::array_t<DatapointIndex>, pybind11::array_t<float>>
ScannNumpy::SearchBatched(const np_row_major_arr<float>& queries, int final_nn,
                          int pre_reorder_nn, int leaves,  int batch_size, bool parallel) {
  if (queries.ndim() != 2)
    throw std::invalid_argument("Queries must be in two-dimensional array");

  vector<float> queries_vec(queries.data(), queries.data() + queries.size());
  auto query_dataset = DenseDataset<float>(queries_vec, queries.shape()[0]);

  pybind11::array_t<unsigned long long int> SOW({static_cast<long>(query_dataset.size() * (leaves + 1)), 1L});
  // pybind11::array_t<unsigned long long int> SOW({static_cast<long>(batch_size * (leaves + 1)), 1L});
  // printf("query_dataset.size() = %d, batch_size = %d\n", query_dataset.size(), batch_size);
  // ADD FOR SOW INFO
  auto sow_ptr = reinterpret_cast<unsigned long long int*>(SOW.request().ptr);                   // ADD FOR SOW INFO
  size_t begin = 0;
  size_t curSize = 0;

  std::vector<NNResultsVector> res(query_dataset.size());
  Status status;
  if (parallel)
    status = scann_.SearchBatchedParallel(query_dataset, MakeMutableSpan(res),
                                          final_nn, pre_reorder_nn, leaves,
                                          batch_size,
                                          sow_ptr,
                                          begin,
                                          curSize);  // [ANNA] batch size, sow_ptr added
  else
    status = scann_.SearchBatched(query_dataset, MakeMutableSpan(res), final_nn,
                                  pre_reorder_nn, leaves);
  RuntimeErrorIfNotOk("Error during search: ", status);

  // if (!res.empty()) final_nn = res.front().size();
  pybind11::array_t<DatapointIndex> indices(
      {static_cast<long>(query_dataset.size()), static_cast<long>(final_nn)});
  pybind11::array_t<float> distances(
      {static_cast<long>(query_dataset.size()), static_cast<long>(final_nn)});
  auto idx_ptr = reinterpret_cast<DatapointIndex*>(indices.request().ptr);
  auto dis_ptr = reinterpret_cast<float*>(distances.request().ptr);
  scann_.ReshapeBatchedNNResult(MakeConstSpan(res), idx_ptr, dis_ptr, final_nn);
  return {SOW, indices, distances};
}

void ScannNumpy::Serialize(std::string path, std::string coarse_path, bool load_coarse, std::string fine_path, bool load_fine) {
  Status status = scann_.Serialize(path, coarse_path, load_coarse, fine_path, load_fine);
  RuntimeErrorIfNotOk("Failed to extract SingleMachineFactoryOptions: ",
                      status);
}

}  // namespace research_scann
