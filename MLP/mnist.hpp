#ifndef mnist_hpp
#define mnist_hpp

#include <stdio.h>
#include <vector>

using namespace std;

namespace mnist {
    typedef struct node {
        vector<double> image;
        int label;
    } mnist_node;
}

#endif /* mnist_hpp */
