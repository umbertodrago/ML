#include "MLP.hpp"
#include <ctime>


MLP::MLP(vector<int> sizes){
    
    num_layers_ = sizes.size();
    layers_size_ = sizes;
    w_shape_.resize(num_layers_ - 1);
    
    for(int i = 0; i < w_shape_.size(); i++){
        w_shape_[i].resize(2);
        w_shape_[i][0] = layers_size_[i + 1];
        w_shape_[i][1] = layers_size_[i];
    }
    
    max_len_ = 0;
    for (int i = 0; i < num_layers_ - 1; i++){
        if (max_len_ < layers_size_[i])
            max_len_ = layers_size_[i];
    }
    
    buff_1_.resize(max_len_);
    buff_2_.resize(max_len_);
    
    unsigned seed = (unsigned)std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator (seed);
    normal_distribution<double> distribution (0.0,1.0);

    weights_.resize(num_layers_ - 1);
    biases_.resize(num_layers_ - 1);
    
    for (int i = 1; i < num_layers_; i++){
        int p_layer_n_neurons = layers_size_[i - 1];
        int c_layer_n_neurons = layers_size_[i];
        
        weights_[i - 1].resize(c_layer_n_neurons * p_layer_n_neurons);
        
        for (int j = 0; j < c_layer_n_neurons * p_layer_n_neurons; j++){
            weights_[i - 1][j] = distribution(generator);
        }
        
        biases_[i - 1].resize(c_layer_n_neurons);
        for (int j = 0; j < c_layer_n_neurons; j++){
            biases_[i - 1][j] = distribution(generator);
        }
    }
}

vector<double> MLP::sigmoid(vector<double> z){
    vector<double> res(z.size());
    for (int i = 0; i < z.size(); i++){
        res[i] = 1 / (1 + exp(-z[i]));
    }
    return res;
}

vector<double> MLP::sigmoid_prime(vector<double> z){
    vector<double> res(z.size());
    double sigi;
    for (int i = 0; i < z.size(); i++){
        sigi = 1 / (1 + exp(-z[i]));
        res[i] = sigi * (1 - sigi);
    }
    return res;
}

void MLP::cost_derivative(vector<double> &a, vector<double> &l, vector<double> &r, int N){
    cblas_dcopy(N, &a[0], 1, &r[0], 1);
    cblas_daxpy(N, -1, &l[0], 1, &r[0], 1);
}


void MLP::BP1(vector<double> &a, vector<double> &y, vector<double> &z, vector<double> & r, int N){
    vector<double> s = sigmoid_prime(z);
    vector<double> rint(N);
    
    cost_derivative(a, y, rint, N);
    cblas_dsbmv(CblasRowMajor, CblasLower, N, 0, 1, &rint[0], 1, &s[0], 1, 0, &r[0], 1);
}

void MLP::BP2(vector<double> &w, int RW, int CW, vector<double> &d, vector<double> &z, vector<double> &r){
    vector<double> s = sigmoid_prime(z);
    vector<double> rint(CW);
    cblas_dgemv(CblasRowMajor, CblasTrans, RW, CW, 1, &w[0], CW, &d[0], 1, 0, &rint[0], 1);
    cblas_dsbmv(CblasRowMajor, CblasLower, CW, 0, 1, &rint[0], 1, &s[0], 1, 0, &r[0], 1);
}

void MLP::BP3(vector<double> &d, vector<double> &r, int N){
    cblas_dcopy(N, &d[0], 1, &r[0], 1);
}

void MLP::BP4(vector<double> &d, int d_rows, int d_cols, vector<double> &at, int at_cols, vector<double> &r){
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, d_rows, at_cols, d_cols, 1, &d[0], d_cols, &at[0], at_cols, 0, &r[0], at_cols);
}


void MLP::feedforward(vector<double> &input, vector<double> &output){
    
    cblas_dcopy(layers_size_[0], &input[0], 1, &buff_1_[0], 1);
    cblas_dcopy(layers_size_[1], &biases_[0][0], 1, &buff_2_[0], 1);
    
    for(int i = 1; i < num_layers_; i++){
        cblas_dgemv(CblasRowMajor, CblasNoTrans, w_shape_[i - 1][0], w_shape_[i - 1][1], 1, &weights_[i - 1][0], w_shape_[i - 1][1], &buff_1_[0], 1, 1, &buff_2_[0], 1);
        
        cblas_dcopy(layers_size_[i], &sigmoid(buff_2_)[0], 1, &buff_1_[0], 1);
        
        if (i < num_layers_ - 1){
            cblas_dcopy(layers_size_[i + 1], &biases_[i - 1][0], 1, &buff_2_[0], 1);
        }
    }
    
    cblas_dcopy(layers_size_[num_layers_ - 1], &buff_1_[0], 1, &output[0], 1);
}

int MLP::evaluate(vector<mnist_node> test_data){
    vector<double> r(layers_size_[num_layers_ - 1]);
    int tot = 0;
    
    for(int i = 0; i < test_data_size_; i++){
        
        feedforward(test_data[i].image, r);
        int i_max = cblas_idamax(layers_size_[num_layers_ - 1], &r[0], 1);
        
        if (i_max == test_data[i].label){
            tot++;
        }
    }
    
    return tot;
}

void MLP::backprop(vector<double> image, int label, vector<vector<double>> &dnw, vector<vector<double>> &dnb){
    
    vector<vector<double>> nabla_b(num_layers_ - 1);
    vector<vector<double>> nabla_w(num_layers_ - 1);
    
    for (int i = 0; i < num_layers_ - 1; i++){
        nabla_b[i].resize(layers_size_[i + 1], 0);
        nabla_w[i].resize(layers_size_[i + 1] * layers_size_[i], 0);
    }
    
    vector<vector<double>> activations(num_layers_);
    activations[0].resize(layers_size_[0]);
    cblas_dcopy(layers_size_[0], &image[0], 1, &activations[0][0], 1);
    
    vector<vector<double>> zs(num_layers_ - 1);
    
    // FEEDFORWARD STEP
    cblas_dcopy(layers_size_[1], &biases_[0][0], 1, &buff_1_[0], 1);
    
    for (int i = 1; i < num_layers_; i++){
        cblas_dgemv(CblasRowMajor, CblasNoTrans, w_shape_[i - 1][0], w_shape_[i - 1][1], 1, &weights_[i - 1][0], w_shape_[i - 1][1], &activations[i - 1][0], 1, 1, &buff_1_[0], 1);
        
        activations[i].resize(layers_size_[i]);
        zs[i-1].resize(layers_size_[i]);
        cblas_dcopy(layers_size_[i], &sigmoid(buff_1_)[0], 1, &activations[i][0], 1);
        cblas_dcopy(layers_size_[i], &buff_1_[0], 1, &zs[i - 1][0], 1);
        
        if (i < num_layers_ - 1){
            cblas_dcopy(layers_size_[i + 1], &biases_[i - 1][0], 1, &buff_1_[0], 1);
        }
    }
    
    activations[num_layers_ - 1].resize(layers_size_.back());
    cblas_dcopy(layers_size_.back(), &sigmoid(buff_1_)[0], 1, &activations[num_layers_ - 1][0], 1);
    
    vector<double> delta(layers_size_.back());
    vector<double> label_v(10);
    label_v[label] = 1;
    
    int n_wb = (int)num_layers_ - 1, nl = (int)num_layers_;
    
    BP1(activations[nl - 1], label_v, zs[n_wb - 1], buff_1_, layers_size_[nl - 1]);
    BP3(buff_1_, nabla_b[n_wb - 1], layers_size_[nl - 1]);
    BP4(buff_1_, layers_size_[nl - 1], 1, activations[nl - 2], layers_size_[nl - 2], nabla_w[n_wb - 1]);
    
    
    for(int i = 2; i < num_layers_; i++){
        BP2(weights_[n_wb - i + 1], w_shape_[n_wb - i + 1][0], w_shape_[n_wb - i + 1][1], buff_1_, zs[n_wb - i], buff_2_);
        cblas_dcopy(layers_size_[n_wb - i + 1], &buff_2_[0], 1, &buff_1_[0], 1);
        BP3(buff_1_, nabla_b[n_wb - i], layers_size_[nl - i]);
        BP4(buff_1_, layers_size_[nl - i], 1, activations[nl - 1 - i], layers_size_[nl - 1 - i], nabla_w[n_wb - i]);
    }
    
    dnw = nabla_w; dnb = nabla_b;
}


void MLP::update_minibatch(vector<mnist_node> minibatch, int mb_size, float eta){
    vector<vector<double>> nb(num_layers_ - 1);
    vector<vector<double>> nw(num_layers_ - 1);
    vector<vector<double>> dnw, dnb;
    
    for (int i = 0; i < num_layers_ - 1; i++){
        nb[i].resize(layers_size_[i + 1], 0);
        nw[i].resize(layers_size_[i + 1] * layers_size_[i], 0);
    }
    
    for (int i = 0; i < mb_size; i++){
        backprop(minibatch[i].image, minibatch[i].label, dnw, dnb);
        
        for (int j = 0; j < num_layers_ - 1; j++){
            cblas_daxpy(layers_size_[j + 1], 1, &dnb[j][0], 1, &nb[j][0], 1);
            cblas_daxpy(layers_size_[j + 1] * layers_size_[j], 1, &dnw[j][0], 1, &nw[j][0], 1);
        }
    }
    
    double alpha = -eta/mb_size;
    for (int j = 0; j < num_layers_ - 1; j++){
        cblas_daxpy(layers_size_[j + 1], alpha, &nb[j][0], 1, &biases_[j][0], 1);
        cblas_daxpy(layers_size_[j + 1] * layers_size_[j], alpha, &nw[j][0], 1, &weights_[j][0], 1);
    }
}


void MLP::SGD(vector<mnist_node> training_set, vector<mnist_node> test_set, size_t minibatch_size, int epochs, float eta){
    size_t training_set_size = training_set.size();
    test_data_size_ = test_set.size();
    
    cout << "training set size: " << training_set_size << endl;
    cout << "test set size: " << test_data_size_ << endl;
    cout << "Start SGD" << endl;
    
    for (int i = 0; i < epochs; i++){
        random_shuffle(training_set.begin(), training_set.end());
        vector<vector<mnist_node>> minibatches;
         
        for (int mb_step = 0; mb_step < training_set_size; mb_step += minibatch_size){
            vector<mnist_node> minibatch;
            for (int j = 0; j < minibatch_size; j++){
                int idx = (mb_step + j) % training_set_size;
                minibatch.push_back(training_set[idx]);
            }
            minibatches.push_back(minibatch);
        }
         
         
        for (int j = 0; j < minibatches.size(); j++){
            update_minibatch(minibatches[j], (int)minibatch_size, eta);
        }
        
        cout << "Epoch " << i << ": " << ((float)evaluate(test_set)/test_data_size_) * 100 << "%" << endl;
    }
}

MLP::~MLP(){}





















