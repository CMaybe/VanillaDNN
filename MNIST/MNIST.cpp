#ifndef VANILLA_MNIST_CPP
#define VANILLA_MNIST_CPP

#include "MNIST/MNIST.hpp"

MNIST::MNIST()
{
    set = "defalut";
}

MNIST::MNIST(std::string _set)
{
    std::ifstream imageInputStream;
    std::ifstream labelInputStream;

    this->set = _set;

    if (this->set == "train")
    {
        imageInputStream.open(this->trainSetFileName[0], 
                std::ios::in | std::ios::binary);
        labelInputStream.open(this->trainSetFileName[1],
                std::ios::in | std::ios::binary);
    }
    else if (this->set == "test")
    {
        imageInputStream.open(this->testSetFileName[0],
                std::ios::in | std::ios::binary);
        labelInputStream.open(this->testSetFileName[1],
                std::ios::in | std::ios::binary);
    }
    
    imageInputStream.seekg(0, imageInputStream.end);
    int imageSize = imageInputStream.tellg();
    imageInputStream.seekg(0, imageInputStream.beg);

    labelInputStream.seekg(0, labelInputStream.end);
    int labelSize = labelInputStream.tellg();
    labelInputStream.seekg(0, labelInputStream.beg);

    for (int i = 0; i < imageSize; i += 784)
    {
        std::vector<char> image;
        
        for (int j = 0; j < 784; j++)
        {
            char pixel;
            imageInputStream.read(&pixel, 1);
            image.push_back(pixel);
        }

        this->images.push_back(image);
    }
    
    for (int i = 0; i < labelSize; i++)
    {
        char label;
        labelInputStream.read(&label, 1);
        this->labels.push_back(label);
    }
}

MNIST::MNIST(const MNIST& rhs)
{
    set = "default";
}

std::vector<std::vector<char>> MNIST::getImages()
{
    return this->images;
}

std::vector<char> MNIST::getLabels()
{
    return this->labels;
}

#endif
