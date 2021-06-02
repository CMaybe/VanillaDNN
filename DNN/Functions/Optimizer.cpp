#ifndef VANILLA_DNN_OPTIMIZER_CPP
#define VANILLA_DNN_OPTIMIZER_CPP

#include "Optimizer.hpp"

Optimizer::Optimizer()
{
	this->learning_rate = 1.0;
	this->momentum = 1.0;
	this->vel = 0.0;
	this->beta1 = 1.0;
	this->bete2 = 1.0;
	this->epsilon = EPSILON;
	this->rho = 1.0;
	this->decay = 1.0;
	this->optimizer = OPTIMIZER::GD;
}

Optimizer::~Optimizer()
{

}

void Optimizer::setGD(float lr)
{
	this->optimizer = OPTIMIZER::GD;
	this->learning_rate = lr;
}

void Optimizer::setSGD(float lr)
{
	this->optimizer = OPTIMIZER::SGD;
	this->learning_rate = lr;
}

void Optimizer::setBGD(float lr)
{
	this->optimizer = OPTIMIZER::BGD;
	this->learning_rate = lr;
}

void Optimizer::setMBGD(float lr)
{
	this->optimizer = OPTIMIZER::MBGD;
	this->learning_rate = lr;
}

void Optimizer::setMomentum(float lr, float _momentum)
{
	this->optimizer = OPTIMIZER::MOMENTUM;
	this->learning_rate = lr;
	this->momentum = _momentum;
}

void Optimizer::setNAG(float lr)
{
	this->optimizer = OPTIMIZER::NAG;
	this->learning_rate = lr;
}

void Optimizer::setNadam(float lr)
{
	this->optimizer = OPTIMIZER::NADAM;
	this->learning_rate = lr;
}

void Optimizer::setAdam(float lr, float _beta1, float _beta2, float _decay)
{
	this->optimizer = OPTIMIZER::ADAM;
	this->learning_rate = lr;
	this->beta1 = _beta1;
	this->bete2 = _beta2;
	this->decay = _decay;
}

void Optimizer::setRMSProp(float lr, float _rho, float _epsilon)
{
	this->optimizer = OPTIMIZER::RMSPROP;
	this->learning_rate = lr;
	this->rho = _rho;
	this->epsilon = _epsilon;
}

void Optimizer::setAdagrad(float lr, float _epsilon)
{
	this->optimizer = OPTIMIZER::ADAGRAD;
	this->learning_rate = lr;
	this->epsilon = _epsilon;
}

void Optimizer::setAdaDelta(float lr)
{
	this->optimizer = OPTIMIZER::ADADELTA;
	this->learning_rate = lr;
}

float Optimizer::getLearningRate()
{
	return this->learning_rate;
}

float Optimizer::getCalculatedWeight(float weight, float dE_dW)
{
	float result = 0.0f;
	switch (this->optimizer)
	{
	case OPTIMIZER::GD: case OPTIMIZER::BGD: case OPTIMIZER::MBGD:
		result = weight - this->learning_rate * dE_dW;
		break;
	case OPTIMIZER::MOMENTUM:
		this->vel = this->momentum * this->vel - (this->learning_rate * dE_dW);

		//case OPTIMIZER::
	default:
		break;
	}
	return result;
}

OPTIMIZER Optimizer::getOptimizer()
{
	return this->optimizer;
}


#endif