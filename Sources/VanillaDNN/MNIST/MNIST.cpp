#ifndef VANILLA_MNIST_CPP
#define VANILLA_MNIST_CPP

#include <VanillaDNN/MNIST/MNIST.hpp>
#include <iostream>
MNIST::MNIST()
{
	set = "defalut";
}

MNIST::MNIST(std::string _path,std::string _set,bool onehot)
{
	
	std::string trainSetFileName[2] = \
	{
		_path + "/train-images-idx3-ubyte",
		_path + "/train-labels-idx1-ubyte"
	};
	std::string testSetFileName[2] = \
	{
		_path + "/t10k-images-idx3-ubyte",
		_path + "/t10k-labels-idx1-ubyte"
	};
	
	std::ifstream imageInputStream;
	std::ifstream labelInputStream;

	this->set = _set;
	if (this->set == "train")
	{
		imageInputStream.open(trainSetFileName[0],
			std::ios::in | std::ios::binary);
		labelInputStream.open(trainSetFileName[1],
			std::ios::in | std::ios::binary);
	}
	else if (this->set == "test")
	{
		imageInputStream.open(testSetFileName[0],
			std::ios::in | std::ios::binary);
		labelInputStream.open(testSetFileName[1],
			std::ios::in | std::ios::binary);
	}

	imageInputStream.seekg(0, imageInputStream.end);
	int imageSize = imageInputStream.tellg();
	imageInputStream.seekg(0, imageInputStream.beg);

	labelInputStream.seekg(0, labelInputStream.end);
	int labelSize = labelInputStream.tellg();
	labelInputStream.seekg(0, labelInputStream.beg);

	char temp[16];
	imageInputStream.read(temp, 16);
	for (int i = 0; i < imageSize; i += 784)
	{
		Vector<float> image;

		for (int j = 0; j < 784; j++)
		{
			char pixel;
			imageInputStream.read(&pixel, 1);
			if(onehot){
				image.push_back(static_cast<float>(static_cast<unsigned char>(pixel!=0)));
			}
			else{
				image.push_back(static_cast<float>(static_cast<unsigned char>(pixel)));
			}
		}
		this->images.push_back(image);
	}

	char temp2[8];
	labelInputStream.read(temp2, 8);

	for (int i = 0; i < labelSize; i++)
	{
		char label;
		labelInputStream.read(&label, 1);
		Vector<float> v(10,0);
		v[label] = 1;
		this->labels.push_back(v);
		
	}
}

MNIST::MNIST(const MNIST& rhs)
{
	set = "default";
}

std::vector<Vector<float>> MNIST::getImages()
{
	return this->images;
}

std::vector<Vector<float>> MNIST::getLabels()
{
	return this->labels;
}

std::string MNIST::getSet()
{
	return this->set;
}

#endif
