#ifndef VANILLA_DNN_OUTPUT_LAYER_CPP
#define VANILLA_DNN_OUTPUT_LAYER_CPP

#include <VanillaDNN/Layers/OutputLayer.hpp>

OutputLayer::OutputLayer(int _nNueron) {
	this->nNueron = _nNueron;
}

OutputLayer::~OutputLayer() {
	//if (this->preLayer != nullptr) delete preLayer;
}


#endif
