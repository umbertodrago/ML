#include "ReadMnist.hpp"

ReadMnist::ReadMnist(){}


int ReadMnist::reverse_int (int i) {
    unsigned char c1, c2, c3, c4;
    
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

vector<mnist_node> ReadMnist::read(string images_path, int& number_of_images, int& image_size, string labels_path, int& number_of_labels) {
    
    ifstream img_file(images_path);
    ifstream lab_file(labels_path);
    
    if(img_file.is_open() && lab_file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;
        
        
        // CHECK FILES TO BE CORRECT
        img_file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverse_int(magic_number);
        if(magic_number != 2051) throw runtime_error("Invalid MNIST image file!");
        
        lab_file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverse_int(magic_number);
        if(magic_number != 2049) throw runtime_error("Invalid MNIST label file!");
        
        
        // READ HEADER
        img_file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverse_int(number_of_images);
        img_file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverse_int(n_rows);
        img_file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverse_int(n_cols);
        
        lab_file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverse_int(number_of_labels);
        
        if(number_of_images != number_of_labels) throw runtime_error("Invalid MNIST files!");
        
        
        // CONSTRUCT OUTPUT DATASET
        image_size = n_rows * n_cols;
        
        if(number_of_images > 50000)
            number_of_images = 50000;
        
        vector<mnist_node> dataset(number_of_images);
        
        for(int i = 0; i < number_of_images; i++) {
            
            vector<double> curr_image(image_size);
            
            for(int j = 0; j < image_size; j++) {
                unsigned char tmp_img = 0;
                img_file.read((char *)&tmp_img, sizeof(tmp_img));
                curr_image[j] = (double)tmp_img / 255;
            }

            dataset[i].image = curr_image;
            
            unsigned char tmp_lab = 0;
            lab_file.read((char *)&tmp_lab, sizeof(tmp_lab));
            dataset[i].label = tmp_lab;
        }
        
        return dataset;
    } else {
        throw runtime_error("Cannot open file `" + images_path + "`!");
    }
}
