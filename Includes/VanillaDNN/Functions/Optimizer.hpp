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
	// Optimizer& operator=(const Optimizer& rhs);
	virtual ~Optimizer();
	float getLearningRate();
	void setLearningRate(float lr);
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
	Momentum(float lr, float _momentum);
	virtual ~Momentum(){};
	
	Matrix<float> getWeightGradient(Matrix<float>& dE_dW) override;
	Vector<float> getBiasGradient(Vector<float>& dE_db) override;
};

class NAG : public Optimizer
{
public:	
	NAG(float lr);
	virtual ~NAG(){};
	
	Matrix<float> getWeightGradient(Matrix<float>& dE_dW) override;
	Vector<float> getBiasGradient(Vector<float>& dE_db) override;
};

class Nadam : public Optimizer
{
public:	
	Nadam(float lr);
	virtual ~Nadam(){};
	
	Matrix<float> getWeightGradient(Matrix<float>& dE_dW) override;
	Vector<float> getBiasGradient(Vector<float>& dE_db) override;
};

class RMSProp : public Optimizer
{
private:
	Matrix<float> G_weight; 
	Vector<float> G_bias; 
	float epsilon;
	float rho;
public:	
	RMSProp(float lr, float _rho, float _epsilon);
	virtual ~RMSProp(){};
	
	Matrix<float> getWeightGradient(Matrix<float>& dE_dW) override;
	Vector<float> getBiasGradient(Vector<float>& dE_db) override;
};

class Adagrad : public Optimizer
{
private:
	Matrix<float> G_weight; 
	Vector<float> G_bias;
	float epsilon;
public:	
	Adagrad(float lr, float _epsilon);
	virtual ~Adagrad(){};
	
	Matrix<float> getWeightGradient(Matrix<float>& dE_dW) override;
	Vector<float> getBiasGradient(Vector<float>& dE_db) override;
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
	Adam(float lr, float _beta1, float _beta2, float _epsilon);
	virtual ~Adam(){};
	
	Matrix<float> getWeightGradient(Matrix<float>& dE_dW) override;
	Vector<float> getBiasGradient(Vector<float>& dE_db) override;
};




#endif