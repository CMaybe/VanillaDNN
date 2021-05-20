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
- [ ] ~~Optimizer~~
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
- [ ] ~~Optimizer handle
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

### Examples
TBA


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
