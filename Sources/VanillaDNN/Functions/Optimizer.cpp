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
	result = dE_dW * this->learning_rate / sqrt(G_weight[_depth].norm() + this->epsilon);
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
	result = dE_db * this->learning_rate / sqrt(G_bias[_depth].norm() + this->epsilon);
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
	result = dE_dW * this->learning_rate / sqrt(G_weight[_depth].norm() + this->epsilon);
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
	result = dE_db * this->learning_rate / sqrt(G_bias[_depth].norm() + this->epsilon);
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

Vector<float> Adam::getBiasGradient(Vector<float>& dE_db, int _depth){
	Vector<float> result;
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

Vector<float> AdaDelta::getBiasGradient(Vector<float>& dE_db, int _depth){
	Vector<float> result;
	return result;
}





#endif