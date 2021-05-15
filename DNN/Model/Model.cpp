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
		cur->dz_dw.resize(cur->preLayer->getNueronCnt());
		cur = cur->preLayer;
	}while(cur!=nullptr);
}

void Model::feed_forward(int idx){
	this->inputLayer->inputNeuron = input_set[idx];
	this->inputLayer->outputNeuron = inputLayer->inputNeuron; 
	for(Layer* layer : Layers){
		layer->inputNeuron = layer->weight * layer->preLayer->outputNeuron + layer->bias;
		layer->outputNeuron = layer->activation(layer->inputNeuron);
	}
	this->outputLayer->outputNeuron = outputLayer->inputNeuron;
	this->output = this->outputLayer->outputNeuron;
	return;
}

void Model::back_propagation(int idx){
	Layer* cur = nullptr;
	Layer* next = nullptr;
	this->input = input_set[idx];
	this->target = target_set[idx];
	//output Layer
	cur = this->outputLayer;
	cur->dE_do = this->loss_diff(this->output,this->target);
	cur->do_dz = cur->activation_diff(cur->inputNeuron);
	cur->dz_dw = cur->preLayer->outputNeuron;
	cur->dE_dz = cur->dE_do * cur->do_dz; //for chain rule
	for(int j=0;j<this->nOutput;j++){
		for(int k = 0;k<this->nOutput;k++){
			cur->dE_dw(j, k) = cur->dE_do[j] * cur->do_dz[j] * cur->dz_dw[k];
			cur->weight(j, k) -= cur->dE_dw(j, k);
		}
	}

	//hidden Layer
	do{
		next = cur;
		cur=cur->preLayer; //dE_dh = sigma(dE_Oi)
		for(int j = 0;j<cur->getNueronCnt();j++){
			float temp = 0;
			for(int k = 0;k < next->getNueronCnt();k++){
				temp += next->dE_do[k] * next->weight(k,j);
			}
			cur->dE_do[j] = temp;
		}
		cur->do_dz = cur->activation_diff(cur->inputNeuron);
		cur->dz_dw = cur->preLayer->outputNeuron;
		cur->dE_dz = cur->dE_do * cur->do_dz;

		//gradient
		for(int j=0;j<this->nOutput;j++){
			for(int k = 0;k<this->nOutput;k++){
				cur->dE_dw(j, k) = cur->dE_do[j] * cur->do_dz[j] * cur->dz_dw[k];
				cur->weight(j, k) -= cur->dE_dw(j, k);
			}
		}
		next = cur;
		cur=cur->preLayer;	
	}while(cur != nullptr);

}


void Model::fit(int _batch,int _epoch){
	this->init();
	this->batch = _batch;
	this->epoch = _epoch;
	for(int i = 0;i < this->epoch;i++){
		for(int j = 0;j < this->batch;j++){
			this->feed_forward(j);
			this->back_propagation(j);	
		}
	}
}

void Model::evaluate(int _batch){
	this->accuracy = 0;
	float acc_sum = 0;
	float error_sum = 0;
	this->batch = _batch;
	for(int i = 0;i<this->batch;i++){
		feed_forward(i);
		if(check_success(i)) acc_sum+=1;
		error_sum += this->loss(output,target_set[i]);
	}
	this->accuracy = acc_sum / _batch;
	this->error = error_sum / _batch;
}

float Model::getAccuracy(){
	return this->accuracy;
}

float Model::getError(){
	return this->error;
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

void Model::setTarget(std::vector<Vector<float>>& _target_set){
	this->target_set = _target_set;
}

int Model::getOutput(){
	int idx = 0,_max=-1e9;
	for(int i = 0;i<this->nOutput;i++){
		if(this->output[i] > _max){
			_max = this->output[i];
			idx = i;
		}
	}
	return idx+1;
}

bool Model::check_success(int idx){
	int ans=0;
	for(int i =0;i<nOutput;i++){
		if(target_set[idx][i] != 0){
			ans = idx+1;	
			break;
		}	
	}
	return getOutput()==ans;
}

#endif


