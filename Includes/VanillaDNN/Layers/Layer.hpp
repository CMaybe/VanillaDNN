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
	virtual void connect(std::shared_ptr<Layer>& cur_layer, std::shared_ptr<Layer>& new_layer) = 0;
	virtual void init(int batch_size,std::unique_ptr<Optimizer>& _optimizer) = 0;
	virtual void setInput(const Matrix<float>& _input, const int& idx) = 0;
	virtual void setError(const Vector<float>& error, const int& idx) = 0;
	virtual void setOptimizer(std::unique_ptr<Optimizer>& _optimizer) = 0;
	
	virtual std::shared_ptr<Layer> getPostLayer() = 0;
	virtual std::shared_ptr<Layer> getPreLayer() = 0;
	virtual Matrix<float> getOutput(const int& idx) { return Matrix<float>(0,0); }
	virtual Matrix<float> getFeedback(const int& idx) { return Matrix<float>(0,0); }
	virtual int getDim() const { return 0; }; 
	
	virtual void setActivation(std::string name) = 0;
	
	
	



};

#endif
