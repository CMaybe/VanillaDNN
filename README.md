# Vanilla-DNN

![License](https://img.shields.io/badge/Licence-MIT-blue.svg)


VanillaDNN is DNN framework using only C++. 

## Key Features

  * C++ based DNN library
  * Dependency-free
  
  
## To-do

### DNN
- [x] ~~Training Func~~
- [x] ~~Testing Func~~
- [X] ~~Feed forward~~
- [X] ~~Back propagation~~
- [ ] Optimizer
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
- [ ] Optimizer handle
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
- [ ] stochastic gradient descent 
- [ ] momentum and Nesterov momentum
- [ ] adagrad
- [ ] rmsprop
- [ ] adam
- [ ] adamax


## Dependencies
Nothing. All you need is a C++ compiler.(GCC/G++, GDB, Clang/Clang++)

## Quick Start

You will need CMake to build the code.

First, clone the code:

```
git clone https://github.com/CMaybe/VanillaDNN.git
```

Second, write your code on `main.cpp` in  `./src` folder 

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
```cpp
#include "Math/Matrix/Matrix.hpp"
#include "Math/Vector/Vector.hpp"
#include "DNN/Layer/Layer.hpp"
#include "DNN/Model/Model.hpp"
#include "DNN/Functions/DNNFunction.hpp"
#include "MNIST/MNIST.hpp"
#include <iostream>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
	//training set
	MNIST training_set("train");
	
	// Get MNIST Data
	std::vector<std::vector<char>> pre_training_images = training_set.getImages();
	std::vector<char> pre_training_labels = training_set.getLabels();
	
	
	// For learning Data
	std::vector<Vector<float>> training_images(pre_training_images.size());
	std::vector<Vector<float>> training_labels(pre_training_labels.size());

	for (int i = 0; i < pre_training_images.size(); i++) {
		training_images[i].resize(784, 0);
		for (int j = 0; j < 784; j++) {
			training_images[i][j] = pre_training_images[i][j] != 0;
		}
	}


	for (int i = 0; i < pre_training_labels.size(); i++) {
		training_labels[i].resize(10, 0);
		training_labels[i][pre_training_labels[i]] = 1;
	}

	//evaluate set
	MNIST evaluate_set("test");
	std::vector<std::vector<char>> pre_evaluate_images = evaluate_set.getImages();
	std::vector<char> pre_evaluate_labels = evaluate_set.getLabels();

	std::vector<Vector<float>> evaluate_images(pre_evaluate_images.size());
	std::vector<Vector<float>> evaluate_labels(pre_evaluate_labels.size());

	for (int i = 0; i < pre_evaluate_images.size(); i++) {
		evaluate_images[i].resize(784, 0);
		for (int j = 0; j < 784; j++) {
			evaluate_images[i][j] = pre_evaluate_images[i][j] != 0;
		}
	}


	for (int i = 0; i < pre_evaluate_labels.size(); i++) {
		evaluate_labels[i].resize(10, 0);
		evaluate_labels[i][pre_evaluate_labels[i]] = 1;
	}



	Model mnist(784, 10);//input : 28 x 28, output 0 ~ 9;
	
	mnist.setOutputFunction(ACTIVATION_FUNCTION::sigmoid);
	mnist.setLoss(LOSS_FUNCTION::mean_squared_error);

	mnist.addLayer(new Layer(384, ACTIVATION_FUNCTION::sigmoid));
	mnist.addLayer(new Layer(128, ACTIVATION_FUNCTION::sigmoid));
	mnist.addLayer(new Layer(32, ACTIVATION_FUNCTION::sigmoid));
	
	mnist.setInput(training_images);
	mnist.setTarget(training_labels);
	mnist.fit(50000, 4, 10); //total, epoch, batch

	mnist.setInput(evaluate_images);
	mnist.setTarget(evaluate_labels);

	std::cout << "training is done" << '\n';

	mnist.evaluate(5000);
	std::cout <<"Accuracy : "<< mnist.getAccuracy() << '\n';


	return 0;
}
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
