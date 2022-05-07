#ifndef VANILLA_DNN_CONV2D_LAYER_HPP
#define VANILLA_DNN_CONV2D_LAYER_HPP

#include <vector>
#include <functional>
#include <memory>
#include <tuple>
#include <VanillaDNN/Layers/Layer.hpp>
#include <VanillaDNN/Math/Matrix/Matrix.hpp>
#include <VanillaDNN/Math/Vector/Vector.hpp>
#include <VanillaDNN/Functions/Functions.hpp>
#include <VanillaDNN/Functions/Optimizer.hpp>

class Conv2DLayer : public Layer {
private:
	int height, width, channel, padding_size, kernel_size, stride;
	std::vector<Matrix<float>> filter;
	std::vector<Vector<float>> bias;

	std::vector<std::vector<Matrix<float>>> batch_weight;
	std::vector<std::vector<Vector<float>>> batch_bias;


	std::vector<Matrix<float>> input;
	std::vector<Matrix<float>> output;

	std::vector<Matrix<float>> dE_dw;
	std::vector<Vector<float>> dE_db;

	std::vector<Matrix<float>> dE_do;
	std::vector<Matrix<float>> dE_dz;


	std::vector<Vector<float>> dz_db;
	std::vector<Matrix<float>> dz_dw;

	std::vector<Matrix<float>> feedback;

public:
	Conv2DLayer();
	Conv2DLayer(const Conv2DLayer& rhs);
	Conv2DLayer(const int& channel, const int& kernel_size, const std::string& _activation);
	Conv2DLayer(const int& channel, const int& kernel_size, const int& stride, const std::string& _activation);
	Conv2DLayer(const int& channel, const int& kernel_size, const int& stride, const int& padding_size, const std::string& _activation);

	virtual ~Conv2DLayer();

	virtual void feed_forward(const int& idx);
	virtual void back_propagation(const int& idx);
	virtual void predict();
	virtual void update();
	virtual void init(const int& batch_size, std::unique_ptr<Optimizer>& _optimizer);
	virtual void setInput(const Matrix<float>& _input, const int& idx);
	virtual void setError(const Vector<float>& error, const int& idx);
	virtual void connect(std::shared_ptr<Layer>& cur_layer, std::shared_ptr<Layer>& new_layer);
	virtual void setOptimizer(std::unique_ptr<Optimizer>& _optimizer);

	virtual const Matrix<float>& getOutput(const int& idx);
	virtual const Matrix<float>& getFeedback(const int& idx);
	virtual int getDim() const;

	std::tuple<int, int, int> getShape() const;



};

#endif
