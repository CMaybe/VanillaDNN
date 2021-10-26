#ifndef VANILLA_MNIST_HPP
#define VANILLA_MNIST_HPP

#include <array>
#include <fstream>
#include <string>
#include <vector>
#include <VanillaDNN/Math/Vector/Vector.hpp>

#define ONE_HOT true

class MNIST
{
private:
	std::string set;
	std::vector<Vector<float>> images;
	std::vector<Vector<float>> labels;


public:
	MNIST();
	MNIST(std::string _path, std::string _set, const int& size = 60000, bool onehot = false);
	MNIST(const MNIST& rhs);

	std::vector<Vector<float>> getImages();
	std::vector<Vector<float>> getLabels();
	std::string getSet();
};

#endif
