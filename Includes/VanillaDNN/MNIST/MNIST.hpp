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
	std::vector<std::vector<float>> images;
	std::vector<std::vector<float>> labels;


public:
	MNIST();
	MNIST(std::string _path, std::string _set,bool onehot = false);
	MNIST(const MNIST& rhs);

	std::vector<std::vector<float>> getImages();
	std::vector<std::vector<float>> getLabels();
	std::string getSet();
};

#endif
