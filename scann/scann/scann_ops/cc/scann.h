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

#ifndef SCANN_SCANN_OPS_CC_SCANN_H_
#define SCANN_SCANN_OPS_CC_SCANN_H_

#include <limits>

#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "google/protobuf/text_format.h"
#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/base/single_machine_factory_scann.h"
#include "scann/data_format/dataset.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/utils/threads.h"

namespace research_scann {

class ScannInterface {
 public:
  Status Initialize(ConstSpan<float> dataset,
                    ConstSpan<int32_t> datapoint_to_token,
                    ConstSpan<uint8_t> hashed_dataset,
                    ConstSpan<int8_t> int8_dataset,
                    ConstSpan<float> int8_multipliers,
                    ConstSpan<float> dp_norms, DatapointIndex n_points,
                    const std::string& artifacts_dir, const std::string& coarse_path, const std::string& fine_path);
  Status Initialize(ScannConfig config, SingleMachineFactoryOptions opts,
                    ConstSpan<float> dataset,
                    ConstSpan<int32_t> datapoint_to_token,
                    ConstSpan<uint8_t> hashed_dataset,
                    ConstSpan<int8_t> int8_dataset,
                    ConstSpan<float> int8_multipliers,
                    ConstSpan<float> dp_norms, DatapointIndex n_points);
  Status Initialize(ConstSpan<float> dataset, ConstSpan<float> train_set, DatapointIndex n_points, DatapointIndex t_points,
                    const std::string& config, const bool& load_coarse, const std::string& coarse_path, const bool& load_fine, const std::string& fine_path, int training_threads);
  Status Initialize(
      shared_ptr<DenseDataset<float>> dataset,
      SingleMachineFactoryOptions opts = SingleMachineFactoryOptions(),
      shared_ptr<DenseDataset<float>> train_set=nullptr);

  Status Search(const DatapointPtr<float> query, NNResultsVector* res,
                int final_nn, int pre_reorder_nn, int leaves,
                unsigned long long int* SOW = nullptr,
                size_t begin = 0,
                size_t curSize = 0,
                int arcm_w = 0) const; // [ANNA] SOW
  Status SearchBatched(const DenseDataset<float>& queries,
                       MutableSpan<NNResultsVector> res, int final_nn,
                       int pre_reorder_nn, int leaves,
                       unsigned long long int* SOW = nullptr,
                       size_t begin = 0,
                       size_t curSize = 0,
                       int arcm_w = 0) const;
  Status SearchBatchedParallel(const DenseDataset<float>& queries,
                               MutableSpan<NNResultsVector> res, int final_nn,
                               int pre_reorder_nn, int leaves,
                               int batch_size,
                               unsigned long long int* SOW = nullptr,
                               size_t begin = 0,
                               size_t curSize = 0,
                               int arcm_w = 0) const;   // [ANNA] batch_size, SOW
  Status Serialize(std::string path, std::string coarse_path, bool load_coarse, std::string fine_path, bool load_fine);
  StatusOr<SingleMachineFactoryOptions> ExtractOptions();

  template <typename T_idx>
  void ReshapeNNResult(const NNResultsVector& res, T_idx* indices,
                       float* distances, int final_nn);
  template <typename T_idx>
  void ReshapeBatchedNNResult(ConstSpan<NNResultsVector> res, T_idx* indices,
                              float* distances, int final_nn);

  bool needs_dataset() const { return scann_->needs_dataset(); }
  const Dataset* dataset() const { return scann_->dataset(); }

  size_t n_points() const { return n_points_; }
  DimensionIndex dimensionality() const { return dimensionality_; }
  const ScannConfig* config() const { return &config_; }

 private:
  size_t n_points_;
  DimensionIndex dimensionality_;
  std::unique_ptr<SingleMachineSearcherBase<float>> scann_;
  ScannConfig config_;

  float result_multiplier_;

  size_t min_batch_size_;
};

template <typename T_idx>
void ScannInterface::ReshapeNNResult(const NNResultsVector& res, T_idx* indices,
                                     float* distances, int final_nn) {
  auto j=0;
  for (const auto& p : res) {
    *(indices++) = static_cast<T_idx>(p.first);
    *(distances++) = result_multiplier_ * p.second;
    j++;
  }
  if(j<final_nn){
    for(;j<final_nn; j++) {
      *(indices++) = static_cast<T_idx>(std::numeric_limits<int>::max());
      *(distances++) = -1;
    }
  }
}

template <typename T_idx>
void ScannInterface::ReshapeBatchedNNResult(ConstSpan<NNResultsVector> res,
                                            T_idx* indices, float* distances, int final_nn) {
  auto i=0;
  for (const auto& result_vec : res) {
      i++;
    // [ANNA] fixed: if search results are less than topk, we pad them with dummy results
    auto j=0;
    for (const auto& pair : result_vec) {
      *(indices++) = static_cast<T_idx>(pair.first);
      *(distances++) = result_multiplier_ * pair.second;
      j++;
    }
    if(j<final_nn){
      for(;j<final_nn; j++) {
        *(indices++) = static_cast<T_idx>(std::numeric_limits<int>::max());
        *(distances++) = -1;
      }
    }
  }
}

}  // namespace research_scann

#endif
