#ifndef VANILLA_DNN_OPTIMIZER_HPP
#define VANILLA_DNN_OPTIMIZER_HPP


#include"Math/Matrix/Matrix.hpp"
#include"Math/Vector/Vector.hpp"

#define EPSILON 1e-6


enum class OPTIMIZER
{
	GD,
	SGD,
	BGD,
	MBGD,
	MOMENTUM,
	NAG,
	NADAM,
	ADAM,
	RMSPROP,
	ADAGRAD,
	ADADELTA
};

class Optimizer
{
private:
	float learning_rate;
	float momentum,vel;
	float beta1, bete2;
	float epsilon;
	float rho;
	float decay;
	int batch_size;
	OPTIMIZER optimizer;

public:
	Optimizer();
	virtual ~Optimizer();
	void setSGD(float lr);  //  Stochastic Gradient Descent
	void setBGD(float lr);  //  Batch Gradient Descent
	void setMBGD(float lr); //  Mini-Batfch Gradient Descent
	void setMomentum(float lr, float _momentum);
	void setAdagrad(float lr, float _epsilon);
	void setNAG(float lr);
	void setNadam(float lr);
	void setAdam(float lr, float _beta1, float _beta2, float _decay);
	void setRMSProp(float lr, float _rho, float _epsilon);
	void setAdaDelta(float lr);
	float getLearningRate();
	float getCalculatedWeight(float weight, float dE_dW);
	OPTIMIZER getOptimizer();

};



#endif