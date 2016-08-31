#ifndef ReadMnist_hpp
#define ReadMnist_hpp


#include <stdio.h>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include "mnist.hpp"

using namespace std;
using namespace mnist;

class ReadMnist{
    
private:
    string test_l_, test_i_, train_l_, train_i_;
    
public:
    ReadMnist();
    int reverse_int (int i);
    void read_mnist();
    
    vector<mnist_node> read(string images_path, int& number_of_images, int& image_size, string labels_path, int& number_of_labels);
    void print(int num);
};


#endif /* ReadMnist_hpp */
