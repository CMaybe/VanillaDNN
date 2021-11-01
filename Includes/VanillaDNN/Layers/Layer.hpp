#ifndef VANILLA_DNN_LAYER_HPP
#define VANILLA_DNN_LAYER_HPP

#include <vector>
#include <functional>
#include <memory>
#include <cstring>
#include <string>
#include <VanillaDNN/Math/Matrix/Matrix.hpp>
#include <VanillaDNN/Math/Vector/Vector.hpp>
#include <VanillaDNN/Functions/Functions.hpp>
#include <VanillaDNN/Functions/Optimizer.hpp>


class Layer {
protected:
	int batch_size;
	std::shared_ptr<Layer> preLayer;
	std::weak_ptr<Layer> postLayer;
	std::unique_ptr<Activation> activation = nullptr;
	std::unique_ptr<Optimizer> optimizer = nullptr;
	
public:
	Layer() { this->batch_size = 0; };
	Layer(const Layer& rhs){ this->batch_size = 0; };
	Layer(const int& dim){ this->batch_size = 0; };
	Layer(const int& dim, Activation *_activation){ this->batch_size = 0; };
	
	virtual ~Layer() {};
	virtual void feed_forward(const int& idx) = 0;
	virtual void back_propagation(const int& idx) = 0;
	virtual void update() = 0;
	virtual void predict() = 0;
	virtual void init(const int& batch_size,std::unique_ptr<Optimizer>& _optimizer) = 0;
	virtual void setError(const Vector<float>& error, const int& idx) = 0;
	
	virtual void setInput(const Matrix<float>& _input, const int& idx) { return; };
	virtual void setInput(const std::vector<Matrix<float>>& _input, const int& idx){ return; };
	
	virtual Matrix<float> getOutput(const int& idx) { return Matrix<float>(0,0); }
	virtual Matrix<float> getFeedback(const int& idx) { return Matrix<float>(0,0); }
	virtual int getDim() const { return 0; }; 
	
	
	
	std::shared_ptr<Layer> getPostLayer() { return this->postLayer.lock(); };
	std::shared_ptr<Layer> getPreLayer() {return this->preLayer; };
	std::string getActivationName() const {return this->activation->getName();}
	void setOptimizer(std::unique_ptr<Optimizer>& _optimizer){ this->optimizer = std::unique_ptr<Optimizer>(_optimizer->copy()); };
	
	void connect(std::shared_ptr<Layer>& cur_layer, std::shared_ptr<Layer>& new_layer) {
		new_layer->preLayer = cur_layer;
		cur_layer->postLayer = new_layer; 
	};
	
	void setActivation(const std::string& name) {
		if("None" == name) this->activation = std::make_unique<Activation>(name);
		else if("sigmoid" == name) this->activation = std::make_unique<Sigmoid>(name);
		else if("hyper_tan" == name) this->activation = std::make_unique<HyperTan>(name);
		else if("relu" == name) this->activation = std::make_unique<ReLU>(name);
		else if("leaky_relu" == name) this->activation = std::make_unique<LeakyReLU>(name);
		else if("soft_max" == name) this->activation = std::make_unique<SoftMax>(name);
		return;
	};
	



};

#endif
