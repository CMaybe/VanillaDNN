#ifndef VANILLA_DNN_DENSE_LAYER_CPP
#define VANILLA_DNN_DENSE_LAYER_CPP

#include <VanillaDNN/Layers/DenseLayer.hpp>

DenseLayer::DenseLayer(){}

DenseLayer::DenseLayer(const int& dim) {
	this->preLayer = nullptr;
	this->postLayer = nullptr;
	this->dim = dim;
	this->weight = Matrix<float>(0,0,0);
	this->bias = Vector<float>(0,0);
}

DenseLayer::DenseLayer(const int& dim, Activation _activation) {
	this->preLayer = nullptr;
	this->postLayer = nullptr;
	this->dim = dim;
	this->setActivation(_activation);
}

DenseLayer::~DenseLayer() {
	//if (this->preLayer != nullptr) delete preLayer;
}

void DenseLayer::back_propagation(const int& idx){
	if(this->preLayer == nullptr) return;
	this->do_dz[idx] = this->activation_diff(this->input[idx]);
	this->dz_dw[idx] = this->preLayer->output[idx];
	if(this->postLayer == nullptr){
		this->dE_dz[idx] = this->dE_do[idx] * this->do_dz[idx];
		this->dE_dw[idx] = this->dE_dz[idx].dot(this->dz_dw[idx].transpose());
		this->dE_db[idx] = this->dE_dz[idx];
	}
	else{
		this->dE_do[idx] = this->postLayer->batch_weight[idx].transpose().dot(postLayer->dE_dz[idx]);
		this->dE_dz[idx] = this->dE_do[idx] * this->do_dz[idx];
		this->dE_dw[idx] = this->dE_dz[idx].dot(this->dz_dw[idx].transpose());
		this->dE_db[idx] = this->dE_dz[idx];
	}
	
	return;

}

void DenseLayer::feed_forward(const int& idx){
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
	}
	this->weight -= (this->optimizer->getWeightGradient(dw)) / this->batch_size;
	this->bias -= (this->optimizer->getBiasGradient(db)) / this->batch_size;
	// this->bias -= db / this->batch_size;
	for(int i = 0;i<this->batch_size;i++){
		this->batch_weight[i] = this->weight;
		this->batch_bias[i] = this->bias;
	}
	return;
}

void DenseLayer::init(int batch_size, Optimizer *_optimizer){
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
	this->setOptimizer(_optimizer);

	return;
}


void DenseLayer::setActivation(Activation _activation) {
	this->activation = std::move(std::bind(_activation, std::placeholders::_1));
	//arg.target<int(*)(int, int)>();
	if (*(_activation.target<Vector<float>(*)(Vector<float>&)>()) == ACTIVATION_FUNCTION::sigmoid) {
		activation_diff = std::bind(DIFF_FUNCTION::sigmoid_diff, std::placeholders::_1);
	}
	else if (*(_activation.target<Vector<float>(*)(Vector<float>&)>()) == ACTIVATION_FUNCTION::hyper_tan) {
		activation_diff = std::bind(DIFF_FUNCTION::hyper_tan_diff, std::placeholders::_1);
	}
	else if (*(_activation.target<Vector<float>(*)(Vector<float>&)>()) == ACTIVATION_FUNCTION::ReLU) {
		activation_diff = std::bind(DIFF_FUNCTION::ReLU_diff, std::placeholders::_1);
	}
	else if (*(_activation.target<Vector<float>(*)(Vector<float>&)>()) == ACTIVATION_FUNCTION::leaky_ReLU) {
		activation_diff = std::bind(DIFF_FUNCTION::leaky_ReLU_diff, std::placeholders::_1);
	}
	else if(*(_activation.target<Vector<float>(*)(Vector<float>&)>()) == ACTIVATION_FUNCTION::soft_max){
		activation_diff = DIFF_FUNCTION::soft_max_diff;
	}
	return;
}


void DenseLayer::setInput(const Vector<float>& _input,const int& idx) {
	this->input[idx] = _input;
}

void DenseLayer::setError(const Vector<float>& error,const int& idx) {
	this->dE_do[idx] = error;
}

void DenseLayer::setOptimizer(Optimizer *_optimizer){
	this->optimizer = _optimizer->copy();
	return;
}

Vector<float> DenseLayer::getOutput(const int& idx){
	return this->output[idx];
}

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