#ifndef VANILLA_DNN_INPUT_LAYER_CPP
#define VANILLA_DNN_INPUT_LAYER_CPP

#include <VanillaDNN/Layers/InputLayer.hpp>

InputLayer::InputLayer(int _nNueron) {
	this->nNueron = _nNueron;
	this->setActivation([](Vector<float>& input) ->{
		return input;
	});
}

InputLayer::~InputLayer() {
	//if (this->preLayer != nullptr) delete preLayer;
}


#endif
