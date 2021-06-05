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


public:
	MNIST();
	MNIST(std::string _path, std::string _set);
	MNIST(const MNIST& rhs);

	std::vector<std::vector<char>> getImages();
	std::vector<char> getLabels();
	std::string getSet();
};

#endif
