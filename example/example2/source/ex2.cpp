#include "Math/Matrix/Matrix.hpp"
#include "Math/Vector/Vector.hpp"
#include "MNIST/MNIST.hpp"

#include <iostream>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
	MNIST mnist(std::string("test"));
	std::vector<std::vector<char>> images = mnist.getImages();
	std::vector<char> labels = mnist.getLabels();
	std::cout << images.size() << std::endl;
	std::cout << labels.size() << std::endl;


	for (int i = 0; i < 1; i++) {
		for (int j = 0; j < 28; j++) {
			for (int k = 0; k < 28; k++) {
				if (images[i][j * 28 + k] != 0)std::cout << '1';
				else std::cout << '0';
			}
			std::cout << '\n';
		}
	}

	return 0;
}