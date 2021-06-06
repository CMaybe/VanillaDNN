#ifndef VANILLA_DNN_OPTIMIZER_HPP
#define VANILLA_DNN_OPTIMIZER_HPP


#include <VanillaDNN/Math/Matrix/Matrix.hpp>
#include <VanillaDNN/Math/Vector/Vector.hpp>

#define EPSILON 1e-6

class Optimizer
{
protected:
	float learning_rate;
public:
	Optimizer();
	virtual ~Optimizer();
	float getLearningRate();
	void setLearningRate(float lr);
	virtual Matrix<float> getWeightGradient(Matrix<float>& dE_dW);
	//virtual Vector<float> getBiasGradient(Matrix<float>& dE_dW);
};

class Momentum : public Optimizer
{
private:
	float momentum;
	Matrix<float> vel;
public:	
	Momentum(float lr, float _momentum);
	virtual ~Momentum(){};
	
	Matrix<float> getWeightGradient(Matrix<float>& dE_dW) override;
};

class NAG : public Optimizer
{
public:	
	NAG(float lr);
	virtual ~NAG(){};
	
	Matrix<float> getWeightGradient(Matrix<float>& dE_dW) override;
};

class Nadam : public Optimizer
{
public:	
	Nadam(float lr);
	virtual ~Nadam(){};
	
	Matrix<float> getWeightGradient(Matrix<float>& dE_dW) override;
};

class Adam : public Optimizer
{
private:
	float beta1, beta2, decay;
public:	
	Adam(float lr, float _beta1, float _beta2, float _decay);
	virtual ~Adam(){};
	
	Matrix<float> getWeightGradient(Matrix<float>& dE_dW) override;
};

class RMSProp : public Optimizer
{
private:
	float epsilon;
	float rho;
public:	
	RMSProp(float lr, float _rho, float _epsilo);
	virtual ~RMSProp(){};
	
	Matrix<float> getWeightGradient(Matrix<float>& dE_dW) override;
};

class Adagrad : public Optimizer
{
private:
	float epsilon;
public:	
	Adagrad(float lr, float _epsilon);
	virtual ~Adagrad(){};
	
	Matrix<float> getWeightGradient(Matrix<float>& dE_dW) override;
};

class AdaDelta : public Optimizer
{
public:	
	AdaDelta(float lr);
	virtual ~AdaDelta(){};
	
	Matrix<float> getWeightGradient(Matrix<float>& dE_dW) override;
};



#endif