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
protected:
	using Activation = std::function<Vector<float>(Vector<float>&)>;
	int batch_size;

public:
	Layer();
	Activation activation, activation_diff;
	Layer *preLayer = nullptr, *postLayer=nullptr;
	Optimizer *optimizer = nullptr;
	std::vector<Vector<float>> input; 
	std::vector<Vector<float>> output;


	virtual ~Layer();
	virtual void feed_forward(int idx) = 0;
	virtual void back_propagation(int idx) = 0;
	virtual void update() = 0;
	virtual void connect(Layer * layer) = 0;
	virtual void init(int batch_size, Optimizer *_optimizer) = 0;
	virtual void setInput(const Vector<float>& _input, const int& idx) = 0;
	virtual void setError(const Vector<float>& error, const int& idx) = 0;
	
	virtual Layer* getPostLayer()=0;
	virtual Layer* getPreLayer()=0;
	virtual Vector<float> getOutput(const int& idx) = 0;
	
	
	void setActivation(Activation _activation);



};

#endif
