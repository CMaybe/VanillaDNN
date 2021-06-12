#ifndef VANILLA_DNN_OPTIMIZER_HPP
#define VANILLA_DNN_OPTIMIZER_HPP


#include <VanillaDNN/Math/Matrix/Matrix.hpp>
#include <VanillaDNN/Math/Vector/Vector.hpp>
#include <vector>
#include <cmath>


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
	virtual Matrix<float> getWeightGradient(Matrix<float>& dE_dW, int _depth = 1);
	virtual Vector<float> getBiasGradient(Vector<float>& dE_db, int _depth = 1);
};

class Momentum : public Optimizer
{
private:
	float momentum;
	std::vector<Matrix<float>> vel_weight;
	std::vector<Vector<float>> vel_bias;
public:	
	Momentum(float lr, float _momentum, int _depth);
	virtual ~Momentum(){};
	
	Matrix<float> getWeightGradient(Matrix<float>& dE_dW, int _depth = 1) override;
	Vector<float> getBiasGradient(Vector<float>& dE_db, int _depth = 1) override;
};

class NAG : public Optimizer
{
public:	
	NAG(float lr);
	virtual ~NAG(){};
	
	Matrix<float> getWeightGradient(Matrix<float>& dE_dW, int _depth = 1) override;
	Vector<float> getBiasGradient(Vector<float>& dE_db, int _depth = 1) override;
};

class Nadam : public Optimizer
{
public:	
	Nadam(float lr);
	virtual ~Nadam(){};
	
	Matrix<float> getWeightGradient(Matrix<float>& dE_dW, int _depth = 1) override;
	Vector<float> getBiasGradient(Vector<float>& dE_db, int _depth = 1) override;
};

class RMSProp : public Optimizer
{
private:
	std::vector<Matrix<float>> G_weight; 
	std::vector<Vector<float>> G_bias; 
	float epsilon;
	float rho;
public:	
	RMSProp(float lr, float _rho, float _epsilon, int _depth);
	virtual ~RMSProp(){};
	
	Matrix<float> getWeightGradient(Matrix<float>& dE_dW, int _depth) override;
	Vector<float> getBiasGradient(Vector<float>& dE_db, int _depth = 1) override;
};

class Adagrad : public Optimizer
{
private:
	std::vector<Matrix<float>> G_weight; 
	std::vector<Vector<float>> G_bias;
	float epsilon;
public:	
	Adagrad(float lr, float _epsilon, int _depth);
	virtual ~Adagrad(){};
	
	Matrix<float> getWeightGradient(Matrix<float>& dE_dW, int _depth) override;
	Vector<float> getBiasGradient(Vector<float>& dE_db, int _depth = 1) override;
};

class Adam : public Optimizer
{
private:
	float beta1, beta2, decay;
public:	
	Adam(float lr, float _beta1, float _beta2, float _decay);
	virtual ~Adam(){};
	
	Matrix<float> getWeightGradient(Matrix<float>& dE_dW, int _depth = 1) override;
	Vector<float> getBiasGradient(Vector<float>& dE_db, int _depth = 1) override;
};




#endif