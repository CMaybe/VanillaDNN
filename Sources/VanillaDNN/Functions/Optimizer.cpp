#ifndef VANILLA_DNN_OPTIMIZER_CPP
#define VANILLA_DNN_OPTIMIZER_CPP

#include <VanillaDNN/Functions/Optimizer.hpp>

Optimizer::Optimizer()
{
	this->learning_rate = 1.0;
}

Optimizer::~Optimizer()
{

}

float Optimizer::getLearningRate()
{
	std::cout<<"lr : "<<&(this->learning_rate)<<'\n';
	return this->learning_rate;
}

void Optimizer::setLearningRate(float lr)
{
	this->learning_rate = lr;
}

Matrix<float> Optimizer::getWeightGradient(Matrix<float>& dE_dW){
	return dE_dW * this->learning_rate;
}

Vector<float> Optimizer::getBiasGradient(Vector<float>& dE_db){
	return dE_db * this->learning_rate;
}


Momentum::Momentum(float lr, float _momentum)
{
	this->learning_rate = lr;
	this->momentum = _momentum;
}

Optimizer* Momentum::copy(){
	Momentum* result = new Momentum(this->learning_rate, this->momentum);
	return result;
}

Matrix<float> Momentum::getWeightGradient(Matrix<float>& dE_dW){
	if(this->vel_weight.get_rows_size() == 0){
		this->vel_weight = dE_dW * this->learning_rate;
	}
	else{
		this->vel_weight = (dE_dW * this->learning_rate) - (this->vel_weight * this->momentum);
	}
	return this->vel_weight;
}

Vector<float> Momentum::getBiasGradient(Vector<float>& dE_db){
	if(this->vel_bias.get_size() == 0){
		this->vel_bias = dE_db * this->learning_rate;
	}
	else{
		this->vel_bias = (dE_db * this->learning_rate) - (this->vel_bias * this->momentum);
	}
	return this->vel_bias;
}


NAG::NAG(float lr)
{
	this->learning_rate = lr;
}

Optimizer* NAG::copy(){
	NAG* result = new NAG(this->learning_rate);
	return result;
}

Matrix<float> NAG::getWeightGradient(Matrix<float>& dE_dW){
	Matrix<float> result;
	return result;
}

Vector<float> NAG::getBiasGradient(Vector<float>& dE_db){
	Vector<float> result;
	return result;
}


Nadam::Nadam(float lr)
{
	this->learning_rate = lr;
}

Optimizer* Nadam::copy(){
	Nadam* result = new Nadam(this->learning_rate);
	return result;
}

Matrix<float> Nadam::getWeightGradient(Matrix<float>& dE_dW){
	Matrix<float> result;
	return result;
}

Vector<float> Nadam::getBiasGradient(Vector<float>& dE_db){
	Vector<float> result;
	return result;
}


Adagrad::Adagrad(float lr, float _epsilon)
{
	this->learning_rate = lr;
	this->epsilon = _epsilon;
}

Optimizer* Adagrad::copy(){
	Adagrad* result = new Adagrad(this->learning_rate, this->epsilon);
	return result;
}

Matrix<float> Adagrad::getWeightGradient(Matrix<float>& dE_dW){
	Matrix<float> result;
	if(this->G_weight.get_rows_size() == 0){
		this->G_weight = dE_dW.square();
	}
	else{
		this->G_weight = this->G_weight + dE_dW.square();
	}
	result = (G_weight + this->epsilon).sqrt().inv() * this->learning_rate;
	for(int i = 0;i<result.get_rows_size();i++){
		for(int j = 0;j < result.get_cols_size();j++){
			result(i,j) *= dE_dW(i,j);
		}
	}
	return result;
}

Vector<float> Adagrad::getBiasGradient(Vector<float>& dE_db){
	Vector<float> result;
	if(this->G_bias.get_size() == 0){
		this->G_bias = dE_db.square();
	}
	else{
		this->G_bias = this->G_bias + dE_db.square();
	}
	result =  (G_bias + this->epsilon).sqrt().inv() * this->learning_rate;
	for(int i = 0;i<result.get_size();i++){
		result(i) *= dE_db(i);
	}
	return result;
}


RMSProp::RMSProp(float lr, float _rho, float _epsilon)
{
	this->learning_rate = lr;
	this->rho = _rho;
	this->epsilon = _epsilon;
}

Optimizer* RMSProp::copy(){
	RMSProp* result = new RMSProp(this->learning_rate, this->rho, this->epsilon);
	return result;
}


Matrix<float> RMSProp::getWeightGradient(Matrix<float>& dE_dW){
	Matrix<float> result;
	if(this->G_weight.get_rows_size() == 0){
		this->G_weight = dE_dW.square() * (1.0f - this->rho);
	}
	else{
		this->G_weight = (this->G_weight * this->rho) + (dE_dW.square() * (1.0f - this->rho));
	}
	result =  (G_weight + this->epsilon).sqrt().inv() * this->learning_rate;
	for(int i = 0;i<result.get_rows_size();i++){
		for(int j = 0;j < result.get_cols_size();j++){
			result(i,j) *= dE_dW(i,j);
		}
	}
	return result;
}

Vector<float> RMSProp::getBiasGradient(Vector<float>& dE_db){
	Vector<float> result;
	if(this->G_bias.get_size() == 0){
		this->G_bias = dE_db.square() * (1.0f - this->rho);
	}
	else{
		this->G_bias = (this->G_bias * this->rho) + (dE_db.square() * (1.0f - this->rho));
	}
	result =  (G_bias + this->epsilon).sqrt().inv() * this->learning_rate;
	for(int i = 0;i<result.get_size();i++){
		result(i) *= dE_db(i);
	}
	return result;
}

Adam::Adam(float lr, float _beta1, float _beta2, float _epsilon)
{
	this->learning_rate = lr;
	this->beta1 = _beta1;
	this->beta2 = _beta2;
	this->epsilon = _epsilon;
}

Optimizer* Adam::copy(){
	Adam* result = new Adam(this->learning_rate, this->beta1, this->beta2, this->epsilon);
	return result;
}

Matrix<float> Adam::getWeightGradient(Matrix<float>& dE_dW){
	Matrix<float> result, m_weight_hat, v_weight_hat;
	beta1_t *= beta1;
	beta2_t *= beta2;
	if(this->m_weight.get_rows_size() == 0){
		this->m_weight = dE_dW * (1.0f - this->beta1);
		this->v_weight = dE_dW.square() * (1.0f - this->beta2);
	}
	else{
		this->m_weight = (this->m_weight * this->beta1) + (dE_dW * (1.0f - this->beta1));
		this->v_weight = (this->v_weight * this->beta2) + (dE_dW.square() * (1.0f - this->beta2));
	}
	m_weight_hat = this->m_weight / (1.0f - this->beta1_t);
	v_weight_hat = this->v_weight / (1.0f - this->beta2_t);
	
	result =  (v_weight_hat.sqrt() + this->epsilon).inv() * this->learning_rate;
	for(int i = 0;i<result.get_rows_size();i++){
		for(int j = 0;j < result.get_cols_size();j++){
			result(i,j) *= m_weight_hat(i,j);
		}
	}
	
	return result;
}

Vector<float> Adam::getBiasGradient(Vector<float>& dE_db){
	Vector<float> result, m_bias_hat, v_bias_hat;
	beta1_t *= beta1;
	beta2_t *= beta2;
	if(this->m_bias.get_size() == 0){
		this->m_bias = dE_db * (1.0f - this->beta1);
		this->v_bias = dE_db.square() * (1.0f - this->beta2);
	}
	else{
		this->m_bias = (this->m_bias * this->beta1) + (dE_db * (1.0f - this->beta1));
		this->v_bias = (this->v_bias * this->beta2) + (dE_db.square() * (1.0f - this->beta2));
	}
	m_bias_hat = this->m_bias / (1.0f - this->beta1_t);
	v_bias_hat = this->v_bias / (1.0f - this->beta2_t);
	result =  (v_bias_hat.sqrt() + this->epsilon).inv() * this->learning_rate;
	for(int i = 0;i<result.get_size();i++){
		result(i) *= m_bias_hat(i);
	}
	return result;
}





#endif