#ifndef VANILLA_DNN_LAYER_CPP
#define VANILLA_DNN_LAYER_CPP

#include "Layer.hpp"


Layer::Layer(){

}

void Layer::setActivation(void* f){
	this->activation = std::move(std::bind(f,this,std::placeholders::_1));
	return;
}


void Layer::setLoss(void* f){
	this->loss = std::move(std::bind(f,this,std::placeholders::_1,std::placeholders::_2));
	return;
}

#endif


