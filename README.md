# Vanilla-DNN

![License](https://img.shields.io/badge/Licence-MIT-blue.svg)


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
- [ ] Inverse
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
- [ ] adam
- [ ] adamax


## Dependencies
Nothing. All you need is a C++ compiler. (GCC/G++, GDB, Clang/Clang++)

## Quick Start

You will need CMake to build the code.

First, clone the code:

```
git clone https://github.com/CMaybe/VanillaDNN.git
```

Second, write your code 

### Build


#### For Linux

```
mkdir build
cd build
cmake ..
make
```

#### For Windows (Visual Studio 2019)

Open with visual studio(2019)

`build` folder will be created automatically.

copy `data` folder to `build` folder(for MNIST)

build on VS

## Examples
#### Mnist

##### Source code
```cpp
#include <VanillaDNN/DNN/Layers/Layer.hpp>
#include <VanillaDNN/DNN/Model/Model.hpp>
#include <VanillaDNN/DNN/Functions/DNNFunction.hpp>
#include <VanillaDNN/MNIST/MNIST.hpp>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
	//training set
	MNIST training_set(MNIST_DATA_LOCATION, "train");
	std::vector<Vector<float>> training_images(training_set.getImages());
	std::vector<Vector<float>> training_labels(training_set.getLabels());

	for (int i = 0; i < training_images.size(); i++) {
		training_images[i] /= 255.0f;
	}

	//evaluate set
	MNIST evaluate_set(MNIST_DATA_LOCATION, "test");
	std::vector<Vector<float>> evaluate_images(evaluate_set.getImages());
	std::vector<Vector<float>> evaluate_labels(evaluate_set.getLabels());

	for (int i = 0; i < evaluate_images.size(); i++) {
		evaluate_images[i] /= 255.0f;
	}

	std::cout<< "Data processing is done!\n";
	Model mnist(784, 10);//input : 28 x 28, output 0 ~ 9;
	
	mnist.setOutputFunction(ACTIVATION_FUNCTION::sigmoid);
	mnist.setLoss(LOSS_FUNCTION::mean_squared_error);

	mnist.addLayer(new Layer(384, ACTIVATION_FUNCTION::sigmoid));
	mnist.addLayer(new Layer(128, ACTIVATION_FUNCTION::sigmoid));
	mnist.addLayer(new Layer(32, ACTIVATION_FUNCTION::sigmoid));
	
	mnist.setLearningRate(0.8f);
	mnist.setInput(training_images);
	mnist.setTarget(training_labels);
	mnist.fit(50000, 10, 40); //total, epoch, batch

	mnist.setInput(evaluate_images);
	mnist.setTarget(evaluate_labels);

	std::cout << "training is done!" << '\n';

	mnist.evaluate(7000);
	std::cout <<"Accuracy : "<< mnist.getAccuracy() << '\n';


	return 0;
}
```
##### CMakeLists.txt
```
cmake_minimum_required(VERSION 3.10)

project(mnist)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_LIST_DIR}/bin)
set(CMAKE_BUILD_TYPE Release)
add_definitions("-std=c++17")

# for std::async
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(VanillaDNN PATHS ../..)


add_executable(mnist
	source/mnist.cpp
	${VanillaDNN_SOURCE_DIR}/VanillaDNN/DNN/Functions/DNNFunction.cpp
	${VanillaDNN_SOURCE_DIR}/VanillaDNN/DNN/Functions/Optimizer.cpp
	${VanillaDNN_SOURCE_DIR}/VanillaDNN/DNN/Layers/Layer.cpp
	${VanillaDNN_SOURCE_DIR}/VanillaDNN/DNN/Model/Model.cpp
	${VanillaDNN_SOURCE_DIR}/VanillaDNN/MNIST/MNIST.cpp
	${VanillaDNN_INCLUDE_DIR}/VanillaDNN/Math/Matrix/Matrix.cpp
	${VanillaDNN_INCLUDE_DIR}/VanillaDNN/Math/Vector/Vector.cpp
)

target_link_libraries(mnist PRIVATE Threads::Threads) # for std::async
target_include_directories(mnist PUBLIC ${VanillaDNN_INCLUDE_DIR})
target_compile_definitions(mnist PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}") # for MNIST
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
