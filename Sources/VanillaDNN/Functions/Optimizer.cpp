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
	return this->learning_rate;
}

void Optimizer::setLearningRate(float lr)
{
	this->learning_rate = lr;
}

Matrix<float> Optimizer::getWeightGradient(Matrix<float>& dE_dW, int _depth){
	return dE_dW * this->learning_rate;
}

Vector<float> Optimizer::getBiasGradient(Vector<float>& dE_db, int _depth){
	return dE_db * this->learning_rate;
}


Momentum::Momentum(float lr, float _momentum, int _depth)
{
	this->learning_rate = lr;
	this->momentum = _momentum;
	this->vel_weight.resize(_depth+1);
	this->vel_bias.resize(_depth+1);
}

Matrix<float> Momentum::getWeightGradient(Matrix<float>& dE_dW, int _depth){
	if(this->vel_weight[_depth].get_rows_size() == 0){
		this->vel_weight[_depth] = dE_dW * this->learning_rate;
	}
	else{
		this->vel_weight[_depth] = (dE_dW * this->learning_rate) - (this->vel_weight[_depth] * this->momentum);
	}
	return this->vel_weight[_depth];
}

Vector<float> Momentum::getBiasGradient(Vector<float>& dE_db, int _depth){
	if(this->vel_bias[_depth].get_size() == 0){
		this->vel_bias[_depth] = dE_db * this->learning_rate;
	}
	else{
		this->vel_bias[_depth] = (dE_db * this->learning_rate) - (this->vel_bias[_depth] * this->momentum);
	}
	return this->vel_bias[_depth];
}


NAG::NAG(float lr)
{
	this->learning_rate = lr;
}

Matrix<float> NAG::getWeightGradient(Matrix<float>& dE_dW, int _depth){
	Matrix<float> result;
	return result;
}

Vector<float> NAG::getBiasGradient(Vector<float>& dE_db, int _depth){
	Vector<float> result;
	return result;
}


Nadam::Nadam(float lr)
{
	this->learning_rate = lr;
}

Matrix<float> Nadam::getWeightGradient(Matrix<float>& dE_dW, int _depth){
	Matrix<float> result;
	return result;
}

Vector<float> Nadam::getBiasGradient(Vector<float>& dE_db, int _depth){
	Vector<float> result;
	return result;
}




Adagrad::Adagrad(float lr, float _epsilon, int _depth)
{
	this->learning_rate = lr;
	this->epsilon = _epsilon;
	this->G_weight.resize(_depth+1);
	this->G_bias.resize(_depth+1);
}

Matrix<float> Adagrad::getWeightGradient(Matrix<float>& dE_dW, int _depth){
	Matrix<float> result;
	if(this->G_weight[_depth].get_rows_size() == 0){
		this->G_weight[_depth] = dE_dW.square();
	}
	else{
		this->G_weight[_depth] = this->G_weight[_depth] + dE_dW.square();
	}
	result = (G_weight[_depth] + this->epsilon).sqrt().inv() * this->learning_rate;
	for(int i = 0;i<result.get_rows_size();i++){
		for(int j = 0;j < result.get_cols_size();j++){
			result(i,j) *= dE_dW(i,j);
		}
	}
	return result;
}

Vector<float> Adagrad::getBiasGradient(Vector<float>& dE_db, int _depth){
	Vector<float> result;
	if(this->G_bias[_depth].get_size() == 0){
		this->G_bias[_depth] = dE_db.square();
	}
	else{
		this->G_bias[_depth] = this->G_bias[_depth] + dE_db.square();
	}
	result =  (G_bias[_depth] + this->epsilon).sqrt().inv() * this->learning_rate;
	for(int i = 0;i<result.get_size();i++){
		result(i) *= dE_db(i);
	}
	return result;
}


RMSProp::RMSProp(float lr, float _rho, float _epsilon, int _depth)
{
	this->learning_rate = lr;
	this->rho = _rho;
	this->epsilon = _epsilon;
	this->G_weight.resize(_depth+1);
	this->G_bias.resize(_depth+1);
}

Matrix<float> RMSProp::getWeightGradient(Matrix<float>& dE_dW, int _depth){
	Matrix<float> result;
	if(this->G_weight[_depth].get_rows_size() == 0){
		this->G_weight[_depth] = dE_dW.square() * (1.0f - this->rho);
	}
	else{
		this->G_weight[_depth] = (this->G_weight[_depth] * this->rho) + (dE_dW.square() * (1.0f - this->rho));
	}
	result =  (G_weight[_depth] + this->epsilon).sqrt().inv() * this->learning_rate;
	for(int i = 0;i<result.get_rows_size();i++){
		for(int j = 0;j < result.get_cols_size();j++){
			result(i,j) *= dE_dW(i,j);
		}
	}
	return result;
}

Vector<float> RMSProp::getBiasGradient(Vector<float>& dE_db, int _depth){
	Vector<float> result;
	if(this->G_bias[_depth].get_size() == 0){
		this->G_bias[_depth] = dE_db.square() * (1.0f - this->rho);
	}
	else{
		this->G_bias[_depth] = (this->G_bias[_depth] * this->rho) + (dE_db.square() * (1.0f - this->rho));
	}
	result =  (G_bias[_depth] + this->epsilon).sqrt().inv() * this->learning_rate;
	for(int i = 0;i<result.get_size();i++){
		result(i) *= dE_db(i);
	}
	return result;
}

Adam::Adam(float lr, float _beta1, float _beta2, float _epsilon, int _depth)
{
	this->learning_rate = lr;
	this->beta1 = _beta1;
	this->beta2 = _beta2;
	this->epsilon = _epsilon;
	this->m_weight.resize(_depth + 1);
	this->v_weight.resize(_depth + 1);
	this->m_bias.resize(_depth + 1);
	this->v_bias.resize(_depth + 1);
}

Matrix<float> Adam::getWeightGradient(Matrix<float>& dE_dW, int _depth){
	Matrix<float> result, m_weight_hat, v_weight_hat;
	beta1_t *= beta1;
	beta2_t *= beta2;
	if(this->m_weight[_depth].get_rows_size() == 0){
		this->m_weight[_depth] = dE_dW * (1.0f - this->beta1);
		this->v_weight[_depth] = dE_dW.square() * (1.0f - this->beta2);
	}
	else{
		this->m_weight[_depth] = (this->m_weight[_depth] * this->beta1) + (dE_dW * (1.0f - this->beta1));
		this->v_weight[_depth] = (this->v_weight[_depth] * this->beta2) + (dE_dW.square() * (1.0f - this->beta2));
	}
	m_weight_hat = this->m_weight[_depth] / (1.0f - this->beta1_t);
	v_weight_hat = this->v_weight[_depth] / (1.0f - this->beta2_t);
	
	result =  (v_weight_hat.sqrt() + this->epsilon).inv() * this->learning_rate;
	for(int i = 0;i<result.get_rows_size();i++){
		for(int j = 0;j < result.get_cols_size();j++){
			result(i,j) *= m_weight_hat(i,j);
		}
	}
	
	return result;
}

Vector<float> Adam::getBiasGradient(Vector<float>& dE_db, int _depth){
	Vector<float> result, m_bias_hat, v_bias_hat;
	beta1_t *= beta1;
	beta2_t *= beta2;
	if(this->m_bias[_depth].get_size() == 0){
		this->m_bias[_depth] = dE_db * (1.0f - this->beta1);
		this->v_bias[_depth] = dE_db.square() * (1.0f - this->beta2);
	}
	else{
		this->m_bias[_depth] = (this->m_bias[_depth] * this->beta1) + (dE_db * (1.0f - this->beta1));
		this->v_bias[_depth] = (this->v_bias[_depth] * this->beta2) + (dE_db.square() * (1.0f - this->beta2));
	}
	m_bias_hat = this->m_bias[_depth] / (1.0f - this->beta1_t);
	v_bias_hat = this->v_bias[_depth] / (1.0f - this->beta2_t);
	result =  (v_bias_hat.sqrt() + this->epsilon).inv() * this->learning_rate;
	for(int i = 0;i<result.get_size();i++){
		result(i) *= m_bias_hat(i);
	}
	return result;
}





#endif