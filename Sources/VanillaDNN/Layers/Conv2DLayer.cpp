#ifndef VANILLA_DNN_CONV2D_LAYER_CPP
#define VANILLA_DNN_CONV2D_LAYER_CPP

#include <VanillaDNN/Layers/Conv2DLayer.hpp>

Conv2DLayer::Conv2DLayer() {

}

Conv2DLayer::Conv2DLayer(const Conv2DLayer& rhs) {

}

Conv2DLayer::Conv2DLayer(const int& channel, const int& kernel_size, const std::string& _activation) {

}

Conv2DLayer::Conv2DLayer(const int& channel, const int& kernel_size, const int& stride, const std::string& _activation) {

}

Conv2DLayer::Conv2DLayer(const int& channel, const int& kernel_size, const int& stride, const int& padding_size, const std::string& _activation) {

}

Conv2DLayer::~Conv2DLayer() {
	//if (this->preLayer != nullptr) delete preLayer;
}

void Conv2DLayer::back_propagation(const int& idx) {
	return;

}

void Conv2DLayer::feed_forward(const int& idx) {
	return;
}

void Conv2DLayer::predict() {
	return;
}

void Conv2DLayer::update() {
	return;
}

void Conv2DLayer::init(const int& batch_size, std::unique_ptr<Optimizer>& _optimizer) {
	return;
}


void Conv2DLayer::setInput(const Matrix<float>& _input, const int& idx) {
	return;
}

void Conv2DLayer::setError(const Vector<float>& error, const int& idx) {
	return;
}


Matrix<float> Conv2DLayer::getFeedback(const int& idx) {
	return this->feedback[idx];
}

Matrix<float> Conv2DLayer::getOutput(const int& idx) {
	return this->output[idx];
}

std::tuple<int, int, int> Conv2DLayer::getShape() const {
	return std::make_tuple(this->channel, this->height, this->width);
}


#endif