#ifndef VANILLA_DNN_DENSE_LAYER_CPP
#define VANILLA_DNN_DENSE_LAYER_CPP

#include <VanillaDNN/Layers/DenseLayer.hpp>

DenseLayer::DenseLayer(){}

DenseLayer::DenseLayer(int _dim) {
	this->preLayer = nullptr;
	this->postLayer = nullptr;
	this->dim = _dim;
	this->weight = Matrix<float>(0,0,0);
	this->bias = Vector<float>(0,0);
}

DenseLayer::DenseLayer(int _dim, Activation _activation) {
	this->preLayer = nullptr;
	this->postLayer = nullptr;
	this->dim = _dim;
	this->setActivation(_activation);
}

DenseLayer::~DenseLayer() {
	//if (this->preLayer != nullptr) delete preLayer;
}

void DenseLayer::back_propagation(int idx){
	if(this->preLayer == nullptr) return;
	this->do_dz[idx] = this->activation_diff(this->input[idx]);
	this->dz_dw[idx] = this->preLayer->output[idx];
	
	if(this->postLayer == nullptr){
		this->dE_db[idx] = dE_do[idx] * this->do_dz[idx];
		this->dE_dw[idx] = this->dE_db[idx].dot(this->dz_dw[idx].transpose());	
	}
	else{
		this->dE_do[idx] = this->postLayer->batch_weight[idx].transpose().dot(postLayer->dE_dz[idx]);
		this->dE_dz[idx] = this->dE_do[idx] * this->do_dz[idx];
		this->dE_dw[idx] += this->dE_dz[idx].dot(this->dz_dw[idx].transpose());	
		this->dE_db[idx] += this->dE_dz[idx];
	}
	
	return;

}

void DenseLayer::feed_forward(int idx){
	if(this->preLayer==nullptr){
		this->output[idx] = this->input[idx];
	}
	else{
		this->input[idx] = this->batch_weight[idx].dot(this->preLayer->output[idx]) + this->batch_bias[idx];
		this->output[idx] = this->activation(this->input[idx]);
	}
	
	return;
}

void DenseLayer::predict(){
	std::cout<<"output size : " << this->output.size()<<'\n';
	if(this->preLayer==nullptr){
		this->output[0] = this->input[0];
	}
	else{
		this->input[0] = this->weight.dot(this->preLayer->output[0]) + this->bias;
		this->output[0] = this->activation(this->input[0]);
	}
	
	return;
}

void DenseLayer::update(){
	if(this->preLayer == nullptr) return;
	Matrix<float> dw(this->dim,this->preLayer->dim,0.0f);
	Vector<float> db(this->dim,0.0f);

	for(int idx = 0;idx < this->batch_size; idx++){
		dw += this->dE_dw[idx];
		db += this->dE_db[idx];
		this->dE_dw[idx].resize(this->dim, this->preLayer->dim, 0);
		this->dE_do[idx].resize(this->dim, 0);

	}
	this->weight -= this->optimizer.getWeightGradient(dw);
	this->bias -= this->optimizer.getBiasGradient(db);
	for(int i = 0;i<this->batch_size;i++){
		this->batch_weight[i] = this->weight;
		this->batch_bias[i] = this->bias;
	}
}

void DenseLayer::init(int batch_size, Optimizer _optimizer){
	this->batch_size = batch_size;

	
	this->input.resize(this->batch_size);
	this->output.resize(this->batch_size);
	if(this->preLayer == nullptr) return;
	
	this->dE_dw.resize(this->batch_size);
	this->dE_db.resize(this->batch_size);
	this->dE_do.resize(this->batch_size);
	this->dE_dz.resize(this->batch_size);
	this->do_dz.resize(this->batch_size);
	this->dz_db.resize(this->batch_size);
	this->dz_dw.resize(this->batch_size);
	
	this->weight.resize(this->dim, this->preLayer->dim);
	this->bias.resize(this->dim, 0);
	
	this->weight.setRandom();
	this->bias.setRandom();
	

	for(int i = 0;i<this->batch_size;i++){
		this->batch_weight.push_back(this->weight);
		this->batch_bias.push_back(this->bias);
		this->dE_dw[i].resize(this->dim, this->preLayer->dim, 0);
		this->dE_do[i].resize(this->dim, 0);
		this->do_dz[i].resize(this->dim, 0);
		this->dz_dw[i].resize(this->preLayer->dim,0);
		this->dE_db[i].resize(this->dim, 0);
		this->dE_dz[i].resize(this->dim, 0);
		this->dz_db[i].resize(this->dim, 0);
	}
	this->optimizer = _optimizer;


	return;
}

void DenseLayer::setInput(const Vector<float>& _input,const int& idx) {
	this->input[idx] = _input;
}

void DenseLayer::setError(const Vector<float>& error,const int& idx) {
	this->dE_do[idx] = error;
}

Vector<float> DenseLayer::getOutput(const int& idx){
	return this->output[idx];
}

// void DenseLayer::setInput(const Matrix<float>& _input, int idx) {
// 	this->input[idx] = _input;
// }

Layer* DenseLayer::getPostLayer(){
	return this->postLayer;
}

Layer* DenseLayer::getPreLayer(){
	return this->preLayer;
}

void DenseLayer::connect(Layer * layer){
	(dynamic_cast<DenseLayer*>(layer))->preLayer = this;
	this->postLayer = dynamic_cast<DenseLayer*>(layer);
	return;
}


#endif