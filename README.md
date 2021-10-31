# Vanilla-DNN

![License](https://img.shields.io/badge/Licence-MIT-blue.svg) ![CMake](https://github.com/CMaybe/VanillaDNN/actions/workflows/cmake.yml/badge.svg) ![CodeQL](https://github.com/CMaybe/VanillaDNN/actions/workflows/codeql-analysis.yml/badge.svg) ![MSCV](https://github.com/CMaybe/VanillaDNN/actions/workflows/msvc-analysis.yml/badge.svg)

VanillaDNN is Dependency-free DNN framework with C++.

## Key Features

*   C++ based Deep Learning framework
*   Dependency-free

## To-do

### DNN

*   \[x] ~~Training Func~~
*   \[x] ~~Testing Func~~
*   \[x] ~~Feed forward~~
*   \[x] ~~Back propagation~~
*   \[x] ~~Optimizer~~
*   \[ ] TBA

***

### Math

#### Matrix

*   \[x] ~~Matrices and matrix operations~~
*   \[x] ~~Matrices and vector operations~~
*   \[x] ~~Matrices and scala operations~~
*   \[x] ~~Transpose~~
*   \[ ] TBA

#### Vector

*   \[x] ~~Vectors and vector operations~~
*   \[x] ~~Vectors and scala operations~~
*   \[x] ~~Transpose~~
*   \[x] ~~Inner product~~
*   \[ ] cross product
*   \[ ] TBA

***

### Model

*   \[x] ~~Loss function handle~~
*   \[x] ~~Output function(activation) handle~~
*   \[x] ~~fit~~
*   \[x] ~~evaluate~~
*   \[x] ~~Layer handle~~
*   \[x] ~~Set input~~
*   \[x] ~~Set target~~
*   \[x] ~~Get accuracy~~
*   \[x] ~~Optimizer handle~~
*   \[ ] TBA

***

### Layer

*   \[x] ~~Activation function handle~~
*   \[x] ~~weight~~
*   \[x] ~~bias~~
*   \[x] ~~Neurons~~
*   \[x] ~~diff factors~~

***

### activation functions

*   \[x] ~~tanh~~
*   \[x] ~~sigmoid~~
*   \[x] ~~softmax~~
*   \[ ] softplus
*   \[ ] softsign
*   \[x] ~~rectified linear(relu)~~
*   \[x] ~~leaky relu~~
*   \[ ] identity
*   \[ ] scaled tanh
*   \[ ] exponential linear units(elu)
*   \[ ] scaled exponential linear units (selu)

### loss functions

*   \[x] ~~mean squared error~~
*   \[x] ~~root mean squared error~~
*   \[x] ~~cross\_entropy\_error~~
*   \[x] ~~binary\_entropy\_error~~
*   \[ ] mean absolute error
*   \[ ] mean absolute error with epsilon range

### optimization algorithms

*   \[x] ~~stochastic gradient descent~~
*   \[x] ~~batch gradient descent~~
*   \[x] ~~mini-batch gradient descent~~
*   \[x] ~~momentum~~
*   \[x] ~~adagrad~~
*   \[x] ~~rmsprop~~
*   \[x] ~~adam~~
*   \[ ] adamax

## Dependencies

Nothing. All you need is a C++ compiler. (GCC/G++, GDB, Clang/Clang++)

## Quick Start

You will need CMake to build the code.

~~~sh
git clone https://github.com/CMaybe/VanillaDNN.git
~~~

<!---->

~~~
mkdir build
cd build
cmake ..
make
~~~

## Examples

### Mnist

~~~
cd example/mnist
mkdir build
cd build
cmake ..
make
~~~

<!---->

~~~
../bin/mnist
~~~

### Source code

~~~cpp
#include <VanillaDNN/Layers/DenseLayer.hpp>
#include <VanillaDNN/Model/Model.hpp>
#include <VanillaDNN/Functions/Functions.hpp>
#include <VanillaDNN/Functions/Optimizer.hpp>
#include <VanillaDNN/MNIST/MNIST.hpp>
#include <iostream>
#include <vector>


int main(int argc, char** argv) {
	//training set
	MNIST training_set(MNIST_DATA_LOCATION, "train", 500);
	std::vector<Vector<float>> training_images(training_set.getImages());
	std::vector<Vector<float>> training_labels(training_set.getLabels());


	for(int i = 0;i<training_images.size();i++){
		training_images[i] /= 255.0f;
	}


	//evaluate set
	MNIST evaluate_set(MNIST_DATA_LOCATION, "test", 100);
	std::vector<Vector<float>> evaluate_images(evaluate_set.getImages());
	std::vector<Vector<float>> evaluate_labels(evaluate_set.getLabels());


	for(int i = 0;i<evaluate_images.size();i++){
		evaluate_images[i] /= 255.0f;
	}

	std::cout<< "mnist loaded!\n";

	Model mnist;//input : 28 x 28, output 0 ~ 9;


	mnist.setLoss(LOSS_FUNCTION::categorical_cross_entropy);
	mnist.addLayer(new DenseLayer(784, "sigmoid"));
	mnist.addLayer(new DenseLayer(256, "sigmoid"));
	mnist.addLayer(new DenseLayer(128, "sigmoid"));
	mnist.addLayer(new DenseLayer(32, "sigmoid"));
	mnist.addLayer(new DenseLayer(10, "soft_max"));

	// mnist.setOptimizer(new Momentum(0.1f,0.9));
	// mnist.setOptimizer(new Adagrad(0.01f,1e-6));
	// mnist.setOptimizer(new RMSProp(0.01f, 0.9, 1e-8)); //lr, _rho, _epsilon, _depth
	mnist.setOptimizer(new Adam(0.01f, 0.9f, 0.999f, 1e-8)); //lr, _rho, _epsilon, _depth

	mnist.fit(training_images, training_labels, 5, 32); //total, epoch, batch


	std::cout << "training is done!" << '\n';

	mnist.evaluate(evaluate_images, evaluate_labels);
	std::cout <<"Accuracy : "<< mnist.getAccuracy() << '\n';


	return 0;
}
~~~

### CMakeLists.txt of Example

~~~
cmake_minimum_required(VERSION 3.10)

project(mnist)

set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Degug)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_LIST_DIR}/bin/${CMAKE_BUILD_TYPE})


find_package(VanillaDNN REQUIRED)
add_definitions("-std=c++17")

add_executable(mnist
	source/mnist.cpp
)
target_link_libraries(mnist ${VanillaDNN_LIBRARY_DIR}/${CMAKE_BUILD_TYPE}/libVanillaDNN.a)
target_compile_definitions(mnist PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")

~~~

## Documentation

TBA

## How To Contribute

Contributions are always welcome, either reporting issues/bugs or forking the repository and then issuing pull requests when you have completed some additional coding that you feel will be beneficial to the main project. If you are interested in contributing in a more dedicated capacity, then please contact me.

## License

<img align="right" src="http://opensource.org/trademarks/opensource/OSI-Approved-License-100x137.png">

The class is licensed under the [MIT License](http://opensource.org/licenses/MIT):

Copyright © 2021 Jaegyeom Kim

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
