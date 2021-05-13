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


void Model::learn(){
	this->init();
	this->feed_forward();
	
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


