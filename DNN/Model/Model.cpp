#ifndef VANILLA_DNN_MODEL_CPP
#define VANILLA_DNN_MODEL_CPP

#include "Model.hpp"


Model::Model(){

}

void Model::setLoss(Loss _loss){
	this->loss = std::move(std::bind(_loss,std::placeholders::_1,std::placeholders::_2));
	return;
}

#endif


