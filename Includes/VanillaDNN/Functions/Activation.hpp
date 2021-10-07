#ifndef VANILLA_DNN_ACTIVATON_HPP
#define VANILLA_DNN_ACTIVATON_HPP


#include <VanillaDNN/Math/Vector/Vector.hpp>
#include <VanillaDNN/Math/Matrix/Matrix.hpp>
#include <string>
#include <cmath>

class Activation {
public:
	Activation(){};
	Activation(std::string name) { this->name = name; };
	virtual const std::string getName() { return this->name; };
	
	
	virtual Vector<float> getActivated(Vector<float>& input){ return input; };
	virtual Matrix<float> getActivated(Matrix<float>& input){ return input; };
	virtual Vector<float> getActivatedDiff(Vector<float>& input){ return input; };
	virtual Matrix<float> getActivatedDiff(Matrix<float>& input){ return input; };
	
	virtual Matrix<float> getActivatedDiff2(Vector<float>& input){ return Matrix<float>(1,1,1); };
	
	virtual ~Activation(){};
protected:
	std:: string name;
	
};

class Sigmoid : public Activation{
public:
	Sigmoid(){};
	Sigmoid(std::string name) {this->name = name;};
	virtual ~Sigmoid(){};
	
	virtual const std::string getName() { return this->name; };
	
	virtual Vector<float> getActivated(Vector<float>& input);
	virtual Matrix<float> getActivated(Matrix<float>& input); 
	virtual Vector<float> getActivatedDiff(Vector<float>& input);
	virtual Matrix<float> getActivatedDiff(Matrix<float>& input);
};

class HyperTan : public Activation{
public:
	HyperTan(){};
	HyperTan(std::string name) {this->name = name;};
	virtual ~HyperTan(){};
	
	virtual const std::string getName() { return this->name; };
	
	virtual Vector<float> getActivated(Vector<float>& input);
	virtual Matrix<float> getActivated(Matrix<float>& input); 
	virtual Vector<float> getActivatedDiff(Vector<float>& input);
	virtual Matrix<float> getActivatedDiff(Matrix<float>& input);
};


class ReLU : public Activation{
public:
	ReLU(){};
	ReLU(std::string name) {this->name = name;};
	virtual ~ReLU(){};
	
	virtual const std::string getName() { return this->name; };
	
	virtual Vector<float> getActivated(Vector<float>& input);
	virtual Matrix<float> getActivated(Matrix<float>& input); 
	virtual Vector<float> getActivatedDiff(Vector<float>& input);
	virtual Matrix<float> getActivatedDiff(Matrix<float>& input);
};


class LeakyReLU : public Activation{
public:
	LeakyReLU(){};
	LeakyReLU(std::string name) {this->name = name;};
	virtual ~LeakyReLU(){};
	
	virtual const std::string getName() { return this->name; };
	
	virtual Vector<float> getActivated(Vector<float>& input);
	virtual Matrix<float> getActivated(Matrix<float>& input); 
	virtual Vector<float> getActivatedDiff(Vector<float>& input);
	virtual Matrix<float> getActivatedDiff(Matrix<float>& input);
};

class SoftMax : public Activation{
public:
	SoftMax(){};
	SoftMax(std::string name) {this->name = name;};
	virtual ~SoftMax(){};
	
	virtual const std::string getName() { return this->name; };
	
	virtual Vector<float> getActivated(Vector<float>& input);
	virtual Matrix<float> getActivatedDiff2(Vector<float>& input);
};




#endif