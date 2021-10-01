#ifndef VANILLA_DNN_LAYER_HPP
#define VANILLA_DNN_LAYER_HPP

#include <vector>
#include <functional>
#include <memory>
#include <cstring>
#include <VanillaDNN/Math/Matrix/Matrix.hpp>
#include <VanillaDNN/Math/Vector/Vector.hpp>
#include <VanillaDNN/Functions/Functions.hpp>
#include <VanillaDNN/Functions/Optimizer.hpp>


class Layer {
private:
	using Activation = std::function<Vector<float>(Vector<float>&)>;
protected:
	int batch_size;

public:
	Layer();
	Layer(const int& dim){};
	Layer(const int& dim, Activation _activation){};

	virtual ~Layer();
	virtual void feed_forward(const int& idx) = 0;
	virtual void back_propagation(const int& idx) = 0;
	virtual void update() = 0;
	virtual void predict() = 0;
	virtual void connect(Layer * layer) = 0;
	virtual void init(int batch_size, Optimizer _optimizer) = 0;
	virtual void setInput(const Vector<float>& _input, const int& idx) = 0;
	virtual void setError(const Vector<float>& error, const int& idx) = 0;
	
	virtual Layer* getPostLayer()=0;
	virtual Layer* getPreLayer()=0;
	virtual Vector<float> getOutput(const int& idx) = 0;
	
	
	



};

#endif
