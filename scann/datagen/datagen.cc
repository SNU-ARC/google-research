#include <iostream>
#include <fstream>
#include <string>
#include <random>

using namespace std;

/* Computation */
float inline squared_sum(float* __restrict__ in, int d) {
  float sum = 0.0f;
  #pragma omp simd reduction(+:sum)
  for (int i = 0; i < d; i++)
    sum += in[i] * in[i];
  return sum;
}
int main(int argc, char *argv[]) {
    
    if(argc != 3 && argc != 4) {
      cout << "Check the command line inputs. 3 inputs are required but " << argc << " inputs are received \n";
      cout << "Usage: ./datagen n d query#\n";
      return 0;
    }
    ofstream ofile;
    ofstream qfile;
    random_device rd;
    mt19937 generator(rd());
    int n = stoi(argv[1]);
    int d = stoi(argv[2]);
    int qsize = stoi(argv[3]);
    //string dist = argv[3];
    //int qsize = 1;
    //if (qsize > n)
    //  qsize = n;

    // Create Distribution
    normal_distribution<> d1{0,1};
    uniform_real_distribution<> d2{-1,1};
    exponential_distribution<> d3(1);
    lognormal_distribution<> d4(1, 1);

    for(int t=0; t<8; t++) {
      string path = "data/";
      string p = "";
      p += to_string(n) + "_" + to_string(d) + "_";
      if (t == 0) 
        p += "normal";
      else if (t == 1)
        p += "uniform";
      else if (t == 2)
        p += "exp";
      else if (t == 3)
        p += "lognormal";
      else if (t == 4)
        p += "cnormal";
      else if (t == 5)
        p += "cuniform";
      else if (t == 6)
        p += "cexp";
      else if (t == 7)
        p += "clognormal";

      cout << "Generating Distribution " << p << "\n";
      ofile.open(path + p + ".txt", ofstream::out | ofstream::trunc);
      qfile.open(path + p + "_query.txt", ofstream::out | ofstream::trunc);
      if(!ofile.is_open() || !qfile.is_open()) {
        cout << "No target directory" << endl;
        exit(1);
      }
      float **data = new float*[n];
      for (int i=0; i<n; i++) {
        data[i] = new float[d];
      }
      float **query = new float*[qsize];
      for (int i=0; i<qsize; i++) {
        query[i] = new float[d];
      }
      // Sample Distribution
      for(int i=0; i<n; i++) {
        for (int j=0; j<d; j++) {
          if(t % 4 ==  0)
            data[i][j] = d1(generator);
          else if(t % 4 == 1)
            data[i][j] = d2(generator);
          else if(t % 4 == 2)
            data[i][j] = d3(generator);
          else if(t % 4 == 3)
            data[i][j] = d4(generator);
        }
      }
      for(int i=0; i<qsize; i++) {
        for (int j=0; j<d; j++) {
          if(t % 4 ==0)
            query[i][j] = d1(generator);
          else if(t % 4 == 1)
            query[i][j] = d2(generator);
          else if(t % 4 == 2)
            query[i][j] = d3(generator);
          else if(t % 4 == 3)
            query[i][j] = d4(generator);
        }
      }

      if (t>=4) {
        /* Normalize */
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
          float norm = sqrt(squared_sum(data[i], d));
          for (int j = 0; j < d; j++) {
            data[i][j] = data[i][j]/norm;
          }
        }
        #pragma omp parallel for
        for (int i = 0; i < qsize; i++) {
          float norm = sqrt(squared_sum(query[i],d));
          for (int j = 0; j < d; j++) {
            query[i][j] = query[i][j]/norm;
          }
        }
      }
      // Print
      for(int i=0; i<n; i++) {
        for (int j=0; j<d; j++) {
          ofile << data[i][j];
          if(j != d-1)
            ofile << " ";
        }
        if(i != n-1)
          ofile << "\n";
      }
      for(int i=0; i<qsize; i++) {
        for (int j=0; j<d; j++) {
          qfile << query[i][j];
          if(j != d-1)
            qfile << " ";
        }
        if(i != qsize-1)
          qfile << "\n";
      }
      for(int i=0; i<qsize; i++)
        delete[] query[i];
      for(int i=0; i<n; i++)
        delete[] data[i];
      delete[] data;
      delete[] query;
      qfile.close();
      ofile.close();
      qfile.close();
      ofile.close();
    }
    return 0;
}