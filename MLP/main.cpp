#include <stdio.h>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include "DownloadFromUrl.hpp"
#include "ReadMnist.hpp"
#include "MLP.hpp"

using namespace std;
using namespace mnist;

// Prototypes
void downloadFiles(string test_labels_name, string test_images_name, string train_labels_name, string train_images_name);

int main(void) {
    
    string test_labels_name = "./t10k-labels-idx1-ubyte";
    string test_images_name = "./t10k-images-idx3-ubyte";
    string train_labels_name = "./train-labels-idx1-ubyte";
    string train_images_name = "./train-images-idx3-ubyte";

    downloadFiles(test_labels_name, test_images_name, train_labels_name, train_images_name);
    
    ReadMnist rmnist;
    int n_lab_test, n_images_test, s_images_test;
    int n_lab_train, n_images_train, s_images_train;
    
    cout << "Reading MNIST files..." << endl;
    vector<mnist_node> test_set = rmnist.read(test_images_name, n_images_test, s_images_test, test_labels_name, n_lab_test);
    vector<mnist_node> train_set = rmnist.read(train_images_name, n_images_train, s_images_train, train_labels_name, n_lab_train);
    
    vector<int> layers(3);
    layers = {784, 30, 10};
    
    cout << "Creating MLP..." << endl;
    MLP mlp(layers);
    mlp.SGD(train_set, test_set, 10, 30, 3.0);
    
    return 0;
}


void downloadFiles(string test_labels_name, string test_images_name, string train_labels_name, string train_images_name){
    DownloadFromUrl dfu;
    
    string test_labels_url = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz";
    string test_images_url = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz";
    string train_labels_url = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz";
    string train_images_url = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz";
    string ctest_labels_name = "./t10k-labels-idx1-ubyte.gz";
    string ctest_images_name = "./t10k-images-idx3-ubyte.gz";
    string ctrain_labels_name = "./train-labels-idx1-ubyte.gz";
    string ctrain_images_name = "./train-images-idx3-ubyte.gz";
    
    
    ifstream f1(test_labels_name.c_str());
    ifstream f2(test_images_name.c_str());
    ifstream f3(train_labels_name.c_str());
    ifstream f4(train_images_name.c_str());
    
    if (!f1.good())
        dfu.download(test_labels_url.c_str(), ctest_labels_name.c_str());
    
    if (!f2.good())
        dfu.download(test_images_url.c_str(), ctest_images_name.c_str());
    
    if (!f3.good())
        dfu.download(train_labels_url.c_str(), ctrain_labels_name.c_str());
    
    if (!f4.good())
        dfu.download(train_images_url.c_str(), ctrain_images_name.c_str());
}
