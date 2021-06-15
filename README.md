# Vanilla-DNN

![License](https://img.shields.io/badge/Licence-MIT-blue.svg) ![CMake](https://github.com/CMaybe/VanillaDNN/actions/workflows/cmake.yml/badge.svg) ![CodeQL](https://github.com/CMaybe/VanillaDNN/actions/workflows/codeql-analysis.yml/badge.svg)


VanillaDNN is DNN framework using only C++. 

## Key Features

  * C++ based Deep Learning framework
  * Dependency-free
  
  
## To-do

### DNN
- [x] ~~Training Func~~
- [x] ~~Testing Func~~
- [X] ~~Feed forward~~
- [X] ~~Back propagation~~
- [x] ~~Optimizer~~
- [ ] TBA
___
### Math
##### Matrix
- [x] ~~Matrices and matrix operations~~
- [x] ~~Matrices and vector operations~~
- [x] ~~Matrices and scala operations~~
- [x] ~~Transpose~~
- [ ] TBA

##### Vector
- [x] ~~Vectors and vector operations~~
- [x] ~~Vectors and scala operations~~
- [x] ~~Transpose~~
- [X] ~~Inner product~~
- [ ] cross product
- [ ] TBA
___

### Model
- [x] ~~Loss function handle~~
- [x] ~~Output function(activation) handle~~
- [x] ~~fit~~
- [x] ~~evaluate~~
- [x] ~~Layer handle~~
- [x] ~~Set input~~
- [x] ~~Set target~~
- [x] ~~Get accuracy~~
- [x] ~~Optimizer handle~~
- [ ] TBA

___
### Layer
- [x] ~~Activation function handle~~
- [x] ~~weight~~
- [x] ~~bias~~
- [x] ~~Neurons~~
- [x] ~~diff factors~~


___
### activation functions
- [x] ~~tanh~~
- [x] ~~sigmoid~~
- [x] ~~softmax~~
- [ ] softplus
- [ ] softsign
- [x] ~~rectified linear(relu)~~
- [x] ~~leaky relu~~
- [ ] identity
- [ ] scaled tanh
- [ ] exponential linear units(elu)
- [ ] scaled exponential linear units (selu)

### loss functions
- [x] ~~mean squared error~~
- [x] ~~root mean squared error~~
- [x] ~~cross_entropy_error~~
- [x] ~~binary_entropy_error~~
- [ ] mean absolute error
- [ ] mean absolute error with epsilon range

### optimization algorithms
- [x] ~~stochastic gradient descent~~
- [x] ~~batch gradient descent~~
- [x] ~~mini-batch gradient descent~~ 
- [x] ~~momentum~~
- [x] ~~adagrad~~
- [x] ~~rmsprop~~
- [x] ~~adam~~
- [ ] adamax


## Dependencies
Nothing. All you need is a C++ compiler. (GCC/G++, GDB, Clang/Clang++)

## Quick Start

You will need CMake to build the code.

First, clone the code:

```
git clone https://github.com/CMaybe/VanillaDNN.git
```
```
cd example/mnist
mkdir build
cd build
cmake ..
make
```



## Examples
#### Mnist

```
git clone https://github.com/CMaybe/VanillaDNN.git
```
```
cd example/mnist
mkdir build
cd build
cmake ..
make
```
```
../bin/mnist
```

##### Source code
```cpp
#include <VanillaDNN/Layers/Layer.hpp>
#include <VanillaDNN/Model/Model.hpp>
#include <VanillaDNN/Functions/Functions.hpp>
#include <VanillaDNN/Functions/Optimizer.hpp>
#include <VanillaDNN/MNIST/MNIST.hpp>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
	#include <VanillaDNN/Layers/Layer.hpp>
#include <VanillaDNN/Model/Model.hpp>
#include <VanillaDNN/Functions/Functions.hpp>
#include <VanillaDNN/Functions/Optimizer.hpp>
#include <VanillaDNN/MNIST/MNIST.hpp>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
	//training set
	MNIST training_set(MNIST_DATA_LOCATION, "train");
	std::vector<Vector<float>> training_images(training_set.getImages());
	std::vector<Vector<float>> training_labels(training_set.getLabels());
	
	for(int i = 0;i<training_images.size();i++){
		training_images[i] /= 255.0f;
	}


	//evaluate set
	MNIST evaluate_set(MNIST_DATA_LOCATION, "test");
	std::vector<Vector<float>> evaluate_images(evaluate_set.getImages());
	std::vector<Vector<float>> evaluate_labels(evaluate_set.getLabels());
	
		
	for(int i = 0;i<evaluate_images.size();i++){
		evaluate_images[i] /= 255.0f;
	}


	std::cout<< "Data processing is done!\n";
	
	Model mnist(784, 10);//input : 28 x 28, output 0 ~ 9;
	
	
	mnist.setLoss(LOSS_FUNCTION::mean_squared_error);
	mnist.addLayer(new Layer(256, ACTIVATION_FUNCTION::sigmoid));
	mnist.addLayer(new Layer(128, ACTIVATION_FUNCTION::sigmoid));
	mnist.addLayer(new Layer(32, ACTIVATION_FUNCTION::sigmoid));
	mnist.setOutputFunction(ACTIVATION_FUNCTION::sigmoid);
	
	// mnist.setOptimizer(new Momentum(0.01,0.9,mnist.getDepth()));
	// mnist.setOptimizer(new Adagrad(0.01f,1e-6,mnist.getDepth()));
	// mnist.setOptimizer(new RMSProp(0.01f, 0.9, 1e-8,mnist.getDepth())); //lr, _rho, _epsilon, _depth
	mnist.setOptimizer(new Adam(0.01f, 0.9f, 0.999f, 1e-8,mnist.getDepth())); //lr, _rho, _epsilon, _depth
	
	mnist.setInput(training_images);
	mnist.setTarget(training_labels);
	mnist.fit(5000, 10, 32); //total, epoch, batch
	
	mnist.setInput(evaluate_images);
	mnist.setTarget(evaluate_labels);

	std::cout << "training is done!" << '\n';

	mnist.evaluate(7000,true);
	std::cout <<"Accuracy : "<< mnist.getAccuracy() << '\n';


	return 0;
}
}
```
##### CMakeLists.txt
```
cmake_minimum_required(VERSION 3.10)

project(mnist)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_LIST_DIR}/bin)
set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Degug)


find_package(VanillaDNN REQUIRED)
add_definitions("-std=c++17")

add_executable(mnist
	source/mnist.cpp
)

target_compile_definitions(mnist PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}") # for mnist
```


## Documentation

TBA

## How To Contribute

Contributions are always welcome, either reporting issues/bugs or forking the repository and then issuing pull requests when you have completed some additional coding that you feel will be beneficial to the main project. If you are interested in contributing in a more dedicated capacity, then please contact me.


## License

<img align="right" src="http://opensource.org/trademarks/opensource/OSI-Approved-License-100x137.png">

The class is licensed under the [MIT License](http://opensource.org/licenses/MIT):

Copyright &copy; 2021 Jaegyeom Kim


Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
