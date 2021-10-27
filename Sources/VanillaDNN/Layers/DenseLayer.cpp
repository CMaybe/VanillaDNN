#ifndef VANILLA_DNN_DENSE_LAYER_CPP
#define VANILLA_DNN_DENSE_LAYER_CPP

#include <VanillaDNN/Layers/DenseLayer.hpp>

DenseLayer::DenseLayer(){
	this->dim = 0;
}

DenseLayer::DenseLayer(const DenseLayer& rhs) {
	this->dim = rhs.dim;
	this->weight = rhs.weight;
	this->bias = rhs.bias;
	this->setActivation(rhs.getActivationName());
}

DenseLayer::DenseLayer(const int& dim) {
	this->dim = dim;
	this->weight = Matrix<float>(0,0,0);
	this->bias = Vector<float>(0,0);
	this->setActivation("None");
}

DenseLayer::DenseLayer(const int& dim, const std::string& _activation) {
	this->dim = dim;
	this->setActivation(_activation);
}

DenseLayer::~DenseLayer() {
	//if (this->preLayer != nullptr) delete preLayer;
}

void DenseLayer::back_propagation(const int& idx){
	if(this->preLayer == nullptr) return;
	if(this->postLayer.expired() == false){
		std::shared_ptr<Layer> post_layer = this->postLayer.lock();
		this->dE_do[idx] = post_layer->getFeedback(idx);
	}

	if(this->activation->getName() == "soft_max"){
		Matrix<float> do_dz = this->activation->getActivatedDiff2(this->input[idx]);
		this->dE_dz[idx] = do_dz.transpose().dot(this->dE_do[idx]);
	}
	else{
		Vector<float> do_dz = this->activation->getActivatedDiff(this->input[idx]);
		this->dE_dz[idx] = this->dE_do[idx] * do_dz;
	}
	this->dz_dw[idx] = this->preLayer->getOutput(idx);
	this->dE_dw[idx] = this->dE_dz[idx].dot(this->dz_dw[idx].transpose());
	this->dE_db[idx] = this->dE_dz[idx];
	this->feedback[idx] = this->batch_weight[idx].transpose().dot(this->dE_dz[idx]);
	return;

}

void DenseLayer::feed_forward(const int& idx){
	if(this->preLayer == nullptr){
		this->output[idx] = this->input[idx];
	}
	else{
		this->input[idx] = this->batch_weight[idx].dot(this->preLayer->getOutput(idx)) + this->batch_bias[idx];
		this->output[idx] = this->activation->getActivated(this->input[idx]);
	}
	return;
}

void DenseLayer::predict(){
	if(this->preLayer == nullptr){
		this->output[0] = this->input[0];
	}
	else{
		this->input[0] = this->weight.dot(this->preLayer->getOutput(0)) + this->bias;
		this->output[0] = this->activation->getActivated(this->input[0]);
	}
	return;
}

void DenseLayer::update(){
	if(this->preLayer == nullptr) return;
	Matrix<float> dw(this->dim,this->preLayer->getDim(),0.0f);
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

void DenseLayer::init(const int& batch_size,std::unique_ptr<Optimizer>& _optimizer){
	this->batch_size = batch_size;
	
	this->input.resize(this->batch_size);
	this->output.resize(this->batch_size);
	if(this->preLayer == nullptr) return;
	this->dE_dw.resize(this->batch_size);
	this->dE_db.resize(this->batch_size);
	this->dE_do.resize(this->batch_size);
	this->dE_dz.resize(this->batch_size);
	this->dz_db.resize(this->batch_size);
	this->dz_dw.resize(this->batch_size);
	this->feedback.resize(this->batch_size);
	
	this->weight.resize(this->dim, this->preLayer->getDim());
	this->bias.resize(this->dim, 0);
	
	this->weight.setRandom();
	this->bias.setRandom();
	
	this->batch_weight.resize(this->batch_size);
	this->batch_bias.resize(this->batch_size);
	
	for(int i = 0;i<this->batch_size;i++){
		this->batch_weight[i] = this->weight;
		this->batch_bias[i] = this->bias;
	}
	
	this->setOptimizer(_optimizer);

	return;
}


void DenseLayer::setActivation(const std::string& name) {
	if("None" == name) this->activation = std::make_unique<Activation>(name);
	else if("sigmoid" == name) this->activation = std::make_unique<Sigmoid>(name);
	else if("hyper_tan" == name) this->activation = std::make_unique<HyperTan>(name);
	else if("relu" == name) this->activation = std::make_unique<ReLU>(name);
	else if("leaky_relu" == name) this->activation = std::make_unique<LeakyReLU>(name);
	else if("soft_max" == name) this->activation = std::make_unique<SoftMax>(name);
	return;
}


void DenseLayer::setInput(const Matrix<float>& _input,const int& idx) {
	this->input[idx] = _input;
}

void DenseLayer::setError(const Vector<float>& error,const int& idx) {
	this->dE_do[idx] = error;
}

void DenseLayer::setOptimizer(std::unique_ptr<Optimizer>& _optimizer){
	this->optimizer = std::unique_ptr<Optimizer>(_optimizer->copy());
	return;
}


Matrix<float> DenseLayer::getFeedback(const int& idx){
	return this->feedback[idx];
}

Matrix<float> DenseLayer::getOutput(const int& idx){
	return this->output[idx];
}

std::shared_ptr<Layer> DenseLayer::getPostLayer(){
	std::shared_ptr<Layer> temp = this->postLayer.lock();
	return temp;
}

std::shared_ptr<Layer> DenseLayer::getPreLayer(){
	return this->preLayer;
}

void DenseLayer::connect(std::shared_ptr<Layer>& cur_layer, std::shared_ptr<Layer>& new_layer){
	(std::dynamic_pointer_cast<DenseLayer>(new_layer))->preLayer = cur_layer;
	this->postLayer = new_layer;
	return;
}

std::string DenseLayer::getActivationName() const{
	return this->activation->getName();
}

int DenseLayer::getDim() const{
	return this->dim;
}


#endif