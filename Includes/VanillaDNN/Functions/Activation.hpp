#ifndef VANILLA_DNN_ACTIVATON_HPP
#define VANILLA_DNN_ACTIVATON_HPP


#include <VanillaDNN/Math/Vector/Vector.hpp>
#include <VanillaDNN/Math/Matrix/Matrix.hpp>
#include <string>
#include <cmath>

class Activation {
public:
	Activation() {};
	explicit Activation(const std::string& name) { this->name = name; };
	virtual const std::string getName() { return this->name; };


	virtual Vector<float> getActivated(Vector<float>& input) { return input; };
	virtual Matrix<float> getActivated(Matrix<float>& input) { return input; };
	virtual Vector<float> getActivatedDiff(Vector<float>& input) { return input; };
	virtual Matrix<float> getActivatedDiff(Matrix<float>& input) { return input; };

	virtual Matrix<float> getActivatedDiff2(Vector<float>& input) { return Matrix<float>(1, 1, 1); };

	virtual ~Activation() {};
protected:
	std::string name;

};

class Sigmoid : public Activation {
public:
	Sigmoid() {};
	explicit Sigmoid(const std::string& name) { this->name = name; };
	virtual ~Sigmoid() {};

	virtual Vector<float> getActivated(Vector<float>& input);
	virtual Matrix<float> getActivated(Matrix<float>& input);
	virtual Vector<float> getActivatedDiff(Vector<float>& input);
	virtual Matrix<float> getActivatedDiff(Matrix<float>& input);

	virtual const std::string getName() { return this->name; };
};

class HyperTan : public Activation {
public:
	HyperTan() {};
	explicit HyperTan(const std::string& name) { this->name = name; };
	virtual ~HyperTan() {};


	virtual Vector<float> getActivated(Vector<float>& input);
	virtual Matrix<float> getActivated(Matrix<float>& input);
	virtual Vector<float> getActivatedDiff(Vector<float>& input);
	virtual Matrix<float> getActivatedDiff(Matrix<float>& input);

	virtual const std::string getName() { return this->name; };
};


class ReLU : public Activation {
public:
	ReLU() {};
	explicit ReLU(const std::string& name) { this->name = name; };
	virtual ~ReLU() {};


	virtual Vector<float> getActivated(Vector<float>& input);
	virtual Matrix<float> getActivated(Matrix<float>& input);
	virtual Vector<float> getActivatedDiff(Vector<float>& input);
	virtual Matrix<float> getActivatedDiff(Matrix<float>& input);

	virtual const std::string getName() { return this->name; };
};


class LeakyReLU : public Activation {
public:
	LeakyReLU() {};
	explicit LeakyReLU(const std::string& name) { this->name = name; };
	virtual ~LeakyReLU() {};


	virtual Vector<float> getActivated(Vector<float>& input);
	virtual Matrix<float> getActivated(Matrix<float>& input);
	virtual Vector<float> getActivatedDiff(Vector<float>& input);
	virtual Matrix<float> getActivatedDiff(Matrix<float>& input);

	virtual const std::string getName() { return this->name; };
};

class SoftMax : public Activation {
public:
	SoftMax() {};
	explicit SoftMax(const std::string& name) { this->name = name; };
	virtual ~SoftMax() {};

	virtual Vector<float> getActivated(Vector<float>& input);
	virtual Matrix<float> getActivatedDiff2(Vector<float>& input);

	virtual const std::string getName() { return this->name; };
};




#endif