#ifndef VANILLA_DNN_MODEL_CPP
#define VANILLA_DNN_MODEL_CPP

#include "Model.hpp"


Model::Model(){

}

Model::~Model(){
	if(this->inputLayer != nullptr) delete this->inputLayer;
	if(this->outputLayer != nullptr) delete this->outputLayer;
	for(Layer* layer : Layers){
		if(layer!=nullptr) delete layer;
	}
	Layers.clear();
}

void Model::setLoss(Loss _loss){
	this->loss = std::move(std::bind(_loss,std::placeholders::_1,std::placeholders::_2));
	if(*(_loss.target<float(*)(Vector<float>,Vector<float>)>()) == LOSS_FUCTION::mean_squared_error){
		loss_diff = std::bind(DIFF_FUNCTION::mean_squared_error_diff,std::placeholders::_1,std::placeholders::_2);
	}
	else if(*(_loss.target<float(*)(Vector<float>,Vector<float>)>()) == LOSS_FUCTION::root_mean_squared_error){
		loss_diff = std::bind(DIFF_FUNCTION::root_mean_squared_error_diff,std::placeholders::_1,std::placeholders::_2);
	}
	else if(*(_loss.target<float(*)(Vector<float>,Vector<float>)>()) == LOSS_FUCTION::cross_entropy_error){
		loss_diff = std::bind(DIFF_FUNCTION::cross_entropy_error_diff,std::placeholders::_1,std::placeholders::_2);
	}
	else if(*(_loss.target<float(*)(Vector<float>,Vector<float>)>()) == LOSS_FUCTION::binary_cross_entropy){
		loss_diff = std::bind(DIFF_FUNCTION::binary_cross_entropy_diff,std::placeholders::_1,std::placeholders::_2);
	}
	return;
}

void Model::init(){
	this->Layers[0]->preLayer = inputLayer;
	outputLayer->preLayer = this->Layers[this->depth-1];
}

void Model::feed_forward(){
	inputLayer->outputNeuron = inputLayer->inputNeuron; 
	for(Layer* layer : Layers){
		layer->inputNeuron = layer->weight * layer->preLayer->outputNeuron + layer->bias;
		layer->outputNeuron = layer->activation(layer->inputNeuron);
	}
	outputLayer->outputNeuron = outputLayer->inputNeuron;
	return;
}

void Model::back_propagation(){
	
}


void Model::learn(){
	this->init();
	this->feed_forward();
	this->back_propagation();
}

void Model::addLayer(Layer* _layer){
	this->Layers.push_back(_layer);
}

void Model::addLayers(std::vector<Layer*>& _layers){
	for(Layer* layer : _layers){
		this->Layers.push_back(layer);
	}
}

#endif


