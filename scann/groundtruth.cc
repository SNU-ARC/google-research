// g++ -shared -o groundtruth.so -fPIC groundtruth.cc

#include <vector>
#include <stdio.h>
#include <utility>
#include <math.h>
#include <algorithm>

#include <iostream>
using namespace std;
bool compare_descending (pair<int, float> first, pair<int, float> second)
{
  if (first.second > second.second)
    return true;
  else
    return false;
}

float inline compute_similarity(float *__restrict__ a, float *__restrict__ b, int d, bool ip=true) {
  float similarity = 0.0f;
  #pragma omp simd reduction(+:similarity)
  if(ip){
  for (int m = 0; m < d; m++)
    similarity += a[m] * b[m];
  }
  else{
    for (int m = 0; m < d; m++)
      similarity += (a[m] - b[m]) * (a[m] - b[m]);
    similarity = -sqrt(similarity);
  }
  return similarity;
}
extern "C" {
  void compute_groundtruth(int num, int N, int D, int qN, float** data, float** query, int** groundtruth, float** groundtruth_simil, bool ip) {
    #pragma omp parallel for num_threads(16)
    for (int j = 0; j < qN; j++) {
        float similarity;
        vector<pair<int, float> > vec_private;
        vec_private.reserve(N);
        for (int i = 0; i < N; i++) {
          similarity = compute_similarity(data[i], query[j], D, ip);
          int idx = num * N + i;
          vec_private.push_back(make_pair(idx, similarity));
        }
        int sortrange = min(1000, N);
        partial_sort(vec_private.begin(), vec_private.begin() + sortrange, vec_private.end(), compare_descending);

        if (num == 0){
          for(int k = 0; k <sortrange; k++) {
            groundtruth[j][k] = vec_private[k].first;
            groundtruth_simil[j][k] = vec_private[k].second;
          }
        // for split cases
        }else{
          vector<pair<int, float> > vec_inter;
          vec_inter.reserve(2*sortrange);

          // load previously saved groundtruth values
          for (int i = 0; i < sortrange; i++){
            vec_inter.push_back(make_pair(groundtruth[j][i], groundtruth_simil[j][i]));
          }

          // add newly computed groundtruth values
          for (int i = 0; i < sortrange; i++){
            vec_inter.push_back(make_pair(vec_private[i].first, vec_private[i].second));
          }

          // sort them together
          partial_sort(vec_inter.begin(), vec_inter.begin() + sortrange, vec_inter.end(), compare_descending);

          // finally add only top-k results to the array
          for (int k = 0; k < sortrange; k++){
            groundtruth[j][k] = vec_inter[k].first;
            groundtruth_simil[j][k] = vec_inter[k].second;
          }

        }
    }
  }
}
