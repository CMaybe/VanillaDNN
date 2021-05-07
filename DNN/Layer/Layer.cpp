#ifndef VANILLA_DNN_LAYER_CPP
#define VANILLA_DNN_LAYER_CPP

#include "Layer.hpp"


Layer::Layer(){

}

void Layer::setActivation(Activation _activation){
	this->activation = std::move(std::bind(_activation,std::placeholders::_1));
	return;
}


#endif


