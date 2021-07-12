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

#ifndef SCANN_SCANN_OPS_CC_SCANN_NPY_H_
#define SCANN_SCANN_OPS_CC_SCANN_NPY_H_

#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>

#include "absl/types/span.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/data_format/dataset.h"
#include "scann/scann_ops/cc/scann.h"

namespace research_scann {

template <typename T>
using np_row_major_arr =
    pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast>;

class ScannNumpy {
 public:
  ScannNumpy(std::optional<const np_row_major_arr<float>> np_dataset,
             std::optional<const np_row_major_arr<int32_t>> datapoint_to_token,
             std::optional<const np_row_major_arr<uint8_t>> hashed_dataset,
             std::optional<const np_row_major_arr<int8_t>> int8_dataset,
             std::optional<const np_row_major_arr<float>> int8_multipliers,
             std::optional<const np_row_major_arr<float>> dp_norms,
             const std::string& artifacts_dir, const std::string& coarse_path, const std::string& fine_path);
  ScannNumpy(const np_row_major_arr<float>& np_dataset, const np_row_major_arr<float>& train_set,
             const std::string& config,
             const bool& load_coarse, const std::string& coarse_path,
             const bool& load_fine, const std::string& fine_path,
             int training_threads);
  std::tuple<pybind11::array_t<float>, pybind11::array_t<float>, pybind11::array_t<DatapointIndex>, pybind11::array_t<float>> Search(
      const np_row_major_arr<float>& query, int l, int final_nn, int pre_reorder_nn,
      int leaves);
  std::tuple<pybind11::array_t<float>, pybind11::array_t<float>, pybind11::array_t<DatapointIndex>, pybind11::array_t<float>>
  SearchBatched(const np_row_major_arr<float>& queries, int l, int final_nn,
                int pre_reorder_nn, int leaves, int batch_size, bool parallel = false);
  void Serialize(std::string path, std::string coarse_dir, bool load_coarse, std::string fine_dir, bool load_fine);

 private:
  ScannInterface scann_;
};

}  // namespace research_scann

#endif
