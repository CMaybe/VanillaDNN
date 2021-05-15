#ifndef VANILLA_MNIST_HPP
#define VANILLA_MNIST_HPP

#include <array>
#include <fstream>
#include <string>
#include <vector>

class MNIST
{
private:
    std::string set;
    std::vector<std::vector<char>> images;
    std::vector<char> labels;

    // t10k-images-idx3-ubyte  t10k-labels-idx1-ubyte  train-images-idx3-ubyte train-labels-idx1-ubyte    
    // image file name idx = 0
    // label file name idx = 1
    std::string trainSetFileName[2] = \
    {
        "../data/train-images-idx3-ubyte",
        "../data/train-labels-idx1-ubyte"
    };
    std::string testSetFileName[2] = \
    {
        "../data/t10k-images-idx3-ubyte",
        "../data/t10k-labels-idx1-ubyte"
    };

public:
    MNIST();
    MNIST(std::string _set);
    MNIST(const MNIST& rhs);

    std::vector<std::vector<char>> getImages();
    std::vector<char> getLabels();
	std::string getSet();
};

#endif
