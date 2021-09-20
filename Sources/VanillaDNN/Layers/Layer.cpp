#ifndef VANILLA_DNN_LAYER_CPP
#define VANILLA_DNN_LAYER_CPP

#include <VanillaDNN/Layers/Layer.hpp>

Layer::Layer(){}

Layer::Layer(int _nNueron) {
	this->nNueron = _nNueron;
	//this->dz_dw.resize(nNueron,0);
}

Layer::Layer(int _nNueron, Activation _activation) {
	this->nNueron = _nNueron;
	this->setActivation(_activation);
}

Layer::~Layer() {
	//if (this->preLayer != nullptr) delete preLayer;
}

void Layer::setActivation(Activation _activation) {
	this->activation = std::move(std::bind(_activation, std::placeholders::_1));
	//arg.target<int(*)(int, int)>();
	if (*(_activation.target<Vector<float>(*)(Vector<float>&)>()) == ACTIVATION_FUNCTION::sigmoid) {
		activation_diff = std::bind(DIFF_FUNCTION::sigmoid_diff, std::placeholders::_1);
	}
	else if (*(_activation.target<Vector<float>(*)(Vector<float>&)>()) == ACTIVATION_FUNCTION::hyper_tan) {
		activation_diff = std::bind(DIFF_FUNCTION::hyper_tan_diff, std::placeholders::_1);
	}
	else if (*(_activation.target<Vector<float>(*)(Vector<float>&)>()) == ACTIVATION_FUNCTION::ReLU) {
		activation_diff = std::bind(DIFF_FUNCTION::ReLU_diff, std::placeholders::_1);
	}
	else if (*(_activation.target<Vector<float>(*)(Vector<float>&)>()) == ACTIVATION_FUNCTION::leaky_ReLU) {
		activation_diff = std::bind(DIFF_FUNCTION::leaky_ReLU_diff, std::placeholders::_1);
	}
	else if(*(_activation.target<Vector<float>(*)(Vector<float>&)>()) == ACTIVATION_FUNCTION::soft_max){
		activation_diff = DIFF_FUNCTION::soft_max_diff;
	}
	return;
}

int Layer::getNueronCnt() {
	return this->nNueron;
}

void Layer::connect(Layer * layer){
	layer->preLayer = this;
	this->postLayer  = layer;
}

#endif