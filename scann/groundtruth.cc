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
  void compute_groundtruth(int N, int D, int qN, float** data, float** query, int** groundtruth, bool ip) { 
    #pragma omp parallel for num_threads(16)
    for (int j = 0; j < qN; j++) {
        float similarity;
        vector<pair<int, float>> vec_private;
        vec_private.reserve(N);
        for(int i = 0; i < N; i++) {
          similarity = compute_similarity(data[i], query[j], D, ip);
          vec_private.push_back(make_pair(i,similarity));
        }
        int sortrange = min(500, N);
        partial_sort(vec_private.begin(), vec_private.begin() + sortrange, vec_private.end(), compare_descending);
        for(int k = 0; k <sortrange; k++) {
          groundtruth[j][k] = vec_private[k].first;
        }
    }
  }
}
