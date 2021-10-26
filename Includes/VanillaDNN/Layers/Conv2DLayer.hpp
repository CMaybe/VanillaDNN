#ifndef VANILLA_DNN_CONV2D_LAYER_HPP
#define VANILLA_DNN_CONV2D_LAYER_HPP

#include <vector>
#include <functional>
#include <memory>
#include <VanillaDNN/Layers/Layer.hpp>
#include <VanillaDNN/Math/Matrix/Matrix.hpp>
#include <VanillaDNN/Math/Vector/Vector.hpp>
#include <VanillaDNN/Functions/Functions.hpp>
#include <VanillaDNN/Functions/Optimizer.hpp>

class Conv2DLayer : public Layer{
	private:
	std::unique_ptr<Activation> activation = nullptr;
	std::unique_ptr<Optimizer> optimizer = nullptr;
	
	int height, width, channel, padding_size, kernel_size, stride;
	Matrix<float> filter;
	Vector<float> bias;
	
	std::vector<Matrix<float>> batch_weight;
	std::vector<Vector<float>> batch_bias;
	
	
	std::vector<Matrix<float>> input; 
	std::vector<Matrix<float>> output;

	std::vector<Matrix<float>> dE_dw;
	std::vector<Vector<float>> dE_db;

	std::vector<Matrix<float>> dE_do;
	std::vector<Matrix<float>> dE_dz;

	
	std::vector<Vector<float>> dz_db;
	std::vector<Matrix<float>> dz_dw;
	
	std::shared_ptr<DenseLayer> preLayer = nullptr, postLayer=nullptr;
	
public:
	Conv2DLayer();
	Conv2DLayer(const Conv2DLayer& rhs);
	Conv2DLayer(const int& channel, const int& kernel_size, std::string _activation);
	Conv2DLayer(const int& channel, const int& kernel_size, const int& stride, std::string _activation);
	Conv2DLayer(const int& channel, const int& kernel_size, const int& stride, const int& padding_size, std::string _activation);
	
	virtual ~Conv2DLayer();

	virtual void feed_forward(const int& idx);
	virtual void back_propagation(const int& idx);
	virtual void predict();
	virtual void update();
	virtual void init(int batch_size,std::unique_ptr<Optimizer>& _optimizer);
	virtual void setInput(const Matrix<float>& _input,const int& idx);
	virtual void setError(const Vector<float>& error,const int& idx);
	virtual void connect(std::shared_ptr<Layer>& cur_layer, std::shared_ptr<Layer>& new_layer);
	virtual void setOptimizer(std::unique_ptr<Optimizer>& _optimizer);
	
	virtual std::shared_ptr<Layer> getPostLayer();
	virtual std::shared_ptr<Layer> getPreLayer();
	virtual Matrix<float> getOutput(const int& idx);
	virtual Matrix<float> getFeedback(const int& idx);
	virtual int getDim() const; 

	virtual void setActivation(std::string name);
	std::string getActivationName() const;



};

#endif
