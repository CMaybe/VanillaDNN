#ifndef VANILLA_DNN_OPTIMIZER_HPP
#define VANILLA_DNN_OPTIMIZER_HPP


#include <VanillaDNN/Math/Matrix/Matrix.hpp>
#include <VanillaDNN/Math/Vector/Vector.hpp>
#include <vector>
#include <iostream>
#include <cmath>


#define EPSILON 1e-6

class Optimizer
{
protected:
	float learning_rate;
public:
	Optimizer();
	Optimizer(float lr);
	
	
	virtual ~Optimizer();
	float getLearningRate();
	void setLearningRate(float lr);
	
	virtual Optimizer* copy();
	virtual Matrix<float> getWeightGradient(Matrix<float>& dE_dW);
	virtual Vector<float> getBiasGradient(Vector<float>& dE_db);
};

class Momentum : public Optimizer
{
private:
	float momentum;
	Matrix<float> vel_weight;
	Vector<float> vel_bias;
public:	
	Momentum(){};
	Momentum(float lr, float _momentum);
	virtual ~Momentum(){};
	
	virtual Optimizer* copy();
	virtual Matrix<float> getWeightGradient(Matrix<float>& dE_dW);
	virtual Vector<float> getBiasGradient(Vector<float>& dE_db);
};

class NAG : public Optimizer
{
public:	
	NAG(){};
	NAG(float lr);
	virtual ~NAG(){};
	
	virtual Optimizer* copy();
	virtual Matrix<float> getWeightGradient(Matrix<float>& dE_dW);
	virtual Vector<float> getBiasGradient(Vector<float>& dE_db);
};

class Nadam : public Optimizer
{
public:	
	Nadam(){};
	Nadam(float lr);
	virtual ~Nadam(){};
	
	virtual Optimizer* copy();
	virtual Matrix<float> getWeightGradient(Matrix<float>& dE_dW);
	virtual Vector<float> getBiasGradient(Vector<float>& dE_db);
};

class Adagrad : public Optimizer
{
private:
	Matrix<float> G_weight; 
	Vector<float> G_bias;
	float epsilon;
public:
	Adagrad(){};
	Adagrad(float lr, float _epsilon);
	virtual ~Adagrad(){};
	
	virtual Optimizer* copy();
	virtual Matrix<float> getWeightGradient(Matrix<float>& dE_dW);
	virtual Vector<float> getBiasGradient(Vector<float>& dE_db);
};

class RMSProp : public Optimizer
{
private:
	Matrix<float> G_weight; 
	Vector<float> G_bias; 
	float epsilon;
	float rho;
public:
	RMSProp(){};
	RMSProp(float lr, float _rho, float _epsilon);
	virtual ~RMSProp(){};
	
	virtual Optimizer* copy();
	virtual Matrix<float> getWeightGradient(Matrix<float>& dE_dW);
	virtual Vector<float> getBiasGradient(Vector<float>& dE_db);
};


class Adam : public Optimizer
{
private:
	float beta1, beta2, epsilon, beta1_t = 1, beta2_t = 1;
	Matrix<float> m_weight; 
	Matrix<float> v_weight;
	Vector<float> m_bias; 
	Vector<float> v_bias; 
public:
	Adam(){};
	Adam(float lr, float _beta1, float _beta2, float _epsilon);
	virtual ~Adam(){};
	
	virtual Optimizer* copy();
	virtual Matrix<float> getWeightGradient(Matrix<float>& dE_dW);
	virtual Vector<float> getBiasGradient(Vector<float>& dE_db);
};




#endif