#ifndef VANILLA_DNN_OPTIMIZER_CPP
#define VANILLA_DNN_OPTIMIZER_CPP

#include <VanillaDNN/DNN/Functions/Optimizer.hpp>

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

Momentum::Momentum(float lr, float _momentum, int _depth)
{
	this->learning_rate = lr;
	this->momentum = _momentum;
	this->vel.resize(_depth+1);
}

Matrix<float> Momentum::getWeightGradient(Matrix<float>& dE_dW, int _depth){
	if(this->vel[_depth].get_rows_size() == 0){
		this->vel[_depth] = dE_dW * this->learning_rate;
	}
	else{
		this->vel[_depth] = (dE_dW * this->learning_rate) - (this->vel[_depth] * this->momentum);
	}
	return this->vel[_depth];
}

NAG::NAG(float lr)
{
	this->learning_rate = lr;
}

Matrix<float> NAG::getWeightGradient(Matrix<float>& dE_dW, int _depth){
	Matrix<float> result;
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

Adam::Adam(float lr, float _beta1, float _beta2, float _decay)
{
	this->learning_rate = lr;
	this->beta1 = _beta1;
	this->beta2 = _beta2;
	this->decay = _decay;
}

Matrix<float> Adam::getWeightGradient(Matrix<float>& dE_dW, int _depth){
	Matrix<float> result;
	return result;
}

RMSProp::RMSProp(float lr, float _rho, float _epsilon)
{
	this->learning_rate = lr;
	this->rho = _rho;
	this->epsilon = _epsilon;
}

Matrix<float> RMSProp::getWeightGradient(Matrix<float>& dE_dW, int _depth){
	Matrix<float> result;
	return result;
}

Adagrad::Adagrad(float lr, float _epsilon)
{
	this->learning_rate = lr;
	this->epsilon = _epsilon;
}

Matrix<float> Adagrad::getWeightGradient(Matrix<float>& dE_dW, int _depth){
	Matrix<float> result;
	return result;
}

AdaDelta::AdaDelta(float lr)
{
	this->learning_rate = lr;
}

Matrix<float> AdaDelta::getWeightGradient(Matrix<float>& dE_dW, int _depth){
	Matrix<float> result;
	return result;
}




#endif