#ifndef VANILLA_DNN_DENSE_LAYER_CPP
#define VANILLA_DNN_DENSE_LAYER_CPP

#include <VanillaDNN/Layers/DenseLayer.hpp>

DenseLayer::DenseLayer(){}

DenseLayer::DenseLayer(int _nNueron) {
	this->nNueron = _nNueron;
	//this->dz_dw.resize(nNueron,0);
}

DenseLayer::DenseLayer(int _nNueron, Activation _activation) {
	this->nNueron = _nNueron;
	this->setActivation(_activation);
}

DenseLayer::~DenseLayer() {
	//if (this->preLayer != nullptr) delete preLayer;
}

void DenseLayer::back_propagation(int idx){
	if(this->preLayer == nullptr) return;
	this->do_dz[idx] = this->activation_diff(this->inputNeuron[idx]);
	this->dz_dw[idx] = this->preLayer->outputNeuron[idx];
	
	if(this->postLayer == nullptr){
		this->dE_db[idx] = this->dE_do[idx] * this->do_dz[idx];
		this->dE_dw[idx] = this->dE_do[idx].dot(this->dz_dw[idx].transpose());	
	}
	else{
		this->dE_do[idx] = this->postLayer->weight.transpose().dot(postLayer->dE_dz[idx]);
		this->dE_dz[idx] = this->postLayer->dE_do[idx] * this->do_dz[idx];
		this->dE_dw[idx] += this->dE_dz[idx].dot(this->dz_dw[idx].transpose());	
		this->dE_db[idx] += this->dE_dz[idx];
	}

}

void DenseLayer::feed_forward(int idx){
	if(this->preLayer==nullptr) return;
	else{
		this->inputNeuron[idx] = this->weight.dot(this->preLayer->outputNeuron[idx]) + this->bias;
		this->outputNeuron[idx] = this->activation(this->inputNeuron[idx]);
	}
}

void DenseLayer::update(const Matrix<float>& dw,const Vector<float>& db){
	if(this->preLayer == nullptr) return;
	this->weight -= dw;
	this->bias -= db;
	
}



#endif