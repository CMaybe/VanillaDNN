#ifndef VANILLA_MNIST_CPP
#define VANILLA_MNIST_CPP

#include <VanillaDNN/MNIST/MNIST.hpp>
#include <iostream>

int char2int(char* p)
{
  return ((p[0] & 0xff) << 24) | ((p[1] & 0xff) << 16) |
         ((p[2] & 0xff) <<  8) | ((p[3] & 0xff) <<  0);
}

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


	char buffer[4];
	imageInputStream.read(buffer, 4);
	
	imageInputStream.read(buffer, 4);
	int imageSize = char2int(buffer);
	
	imageInputStream.read(buffer, 4);
	int rows = char2int(buffer);
	
	imageInputStream.read(buffer, 4);
	int cols = char2int(buffer);
	

	for (int i = 0; i < imageSize; i++)
	{
		Vector<float> image;
		char pixel;
		for (int j = 0; j < rows * cols; j++)
		{
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
	
	labelInputStream.read(buffer, 4);
	
	labelInputStream.read(buffer, 4);
	int labelSize = char2int(buffer);

	for (int i = 0; i < labelSize; i++)
	{
		char label;
		labelInputStream.read(&label, 1);
		Vector<float> v(10,0);
		v[label] = 1;
		this->labels.push_back(v);
		
	}
	
	
	labelInputStream.close();
	imageInputStream.close();
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
