#ifndef VANILLA_DNN_MODEL_CPP
#define VANILLA_DNN_MODEL_CPP

#include "Model.hpp"


Model::Model(){

}

Model::Model(int _nInput,int _nOutput){
	this->depth = 0;
	this->nInput = _nInput;
	this->nOutput = _nOutput;
	this->inputLayer = new Layer(_nInput);
	this->outputLayer = new Layer(_nOutput);
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
	inputLayer->preLayer = nullptr;
	outputLayer->preLayer = this->Layers[this->depth-1];
	Layer * cur = outputLayer;
	do{
		cur->weight.resize(cur->getNueronCnt(),cur->preLayer->getNueronCnt(),1.0);
		cur = cur->preLayer;
	}while(cur!=nullptr);
}

void Model::feed_forward(){
	this->inputLayer->outputNeuron = inputLayer->inputNeuron; 
	for(Layer* layer : Layers){
		layer->inputNeuron = layer->weight * layer->preLayer->outputNeuron + layer->bias;
		layer->outputNeuron = layer->activation(layer->inputNeuron);
	}
	this->outputLayer->outputNeuron = outputLayer->inputNeuron;
	this->output = this->outputLayer->outputNeuron;
	return;
}

void Model::back_propagation(){
	
	
	//outputLayer
	
}


void Model::learn(int _batch,int _epoch){
	this->init();
	this->batch = _batch;
	this->epoch = _epoch;
	for(int i = 0;i < this->epoch;i++){
		this->feed_forward();
		this->back_propagation();	
	}
}

void Model::addLayer(Layer* _layer){
	this->Layers.push_back(_layer);
	if(Layers.size()==1){
		Layers[0]->preLayer = inputLayer;
		outputLayer->preLayer = Layers[0];
	}
	else{
		outputLayer->preLayer = Layers[Layers.size()-1];
		Layers[Layers.size()-1]->preLayer = Layers[Layers.size()-2];
	}
	
	return;
}

void Model::addLayers(std::vector<Layer*>& _layers){
	this->depth += _layers.size();
	for(Layer* layer : _layers){
		this->Layers.push_back(layer);
		if(Layers.size()!=1){
			Layers[Layers.size()-1]->preLayer = Layers[Layers.size()-2]; 
		}
	}
	if(Layers.size()==_layers.size()){
		Layers[0]->preLayer = inputLayer;
	}
	else{
		outputLayer->preLayer = Layers[depth-1];
		Layers[depth-1]->preLayer = Layers[depth-2];
	}
	outputLayer->preLayer = Layers[depth-1];
	
	return;
}

void Model::setInput(std::vector<Vector<float>>& _input_set){
	this->input_set = _input_set;
}

void Model::setOutput(std::vector<Vector<float>>& _target_set){
	this->target_set = _target_set;
}

#endif


