#ifndef MLP_hpp
#define MLP_hpp

#include <stdio.h>
#include <vector>
#include <random>
#include <iostream>
#include "mnist.hpp"
#include <Accelerate/Accelerate.h>

using namespace std;
using namespace mnist;


class MLP{
private:
    vector<vector<double>> biases_;
    vector<vector<double>> weights_;
    vector<vector<int>> w_shape_;
    int max_len_;
    vector<double> buff_1_, buff_2_;
    size_t num_layers_;
    size_t test_data_size_;
    
    clock_t begin;
    clock_t end;
    double elapsed_secs;
    
    vector<int> layers_size_;
    
    void update_minibatch(vector<mnist_node> minibatch, int mb_size, float eta);
    void backprop(vector<double> image, int label, vector<vector<double>> &dnw, vector<vector<double>> &dnb);
    
    void BP1(vector<double> &a, vector<double> &y, vector<double> &z, vector<double> &r, int N);
    void BP2(vector<double> &w, int RW, int CW, vector<double> &d, vector<double> &z, vector<double> &r);
    void BP3(vector<double> &d, vector<double> &r, int N);
    void BP4(vector<double> &d, int d_rows, int d_cols, vector<double> &at, int at_cols, vector<double> &r);
    vector<double> sigmoid(vector<double> z);
    vector<double> sigmoid_prime(vector<double> z);
    void cost_derivative(vector<double> &a, vector<double> &l, vector<double> &r, int N);
    
    void feedforward(vector<double> &input, vector<double> &output);
    int evaluate(vector<mnist_node> test_data);
public:
    MLP(vector<int> sizes);
    ~MLP();
    
    void SGD(vector<mnist_node> training_set, vector<mnist_node> test_set, size_t minibatch_size, int epochs, float eta);
};

#endif /* MLP_hpp */
