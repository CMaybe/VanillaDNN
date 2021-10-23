#ifndef VANILLA_DNN_DENSE_LAYER_HPP
#define VANILLA_DNN_DENSE_LAYER_HPP

#include <vector>
#include <functional>
#include <memory>
#include <VanillaDNN/Layers/Layer.hpp>
#include <VanillaDNN/Math/Matrix/Matrix.hpp>
#include <VanillaDNN/Math/Vector/Vector.hpp>
#include <VanillaDNN/Functions/Functions.hpp>
#include <VanillaDNN/Functions/Optimizer.hpp>




class DenseLayer : public Layer{
	private:
	std::unique_ptr<Activation> activation = nullptr;
	std::unique_ptr<Optimizer> optimizer = nullptr;
	
	int dim;
	Matrix<float> weight;
	Vector<float> bias;
	
	std::vector<Matrix<float>> batch_weight;
	std::vector<Vector<float>> batch_bias;
	
	
	std::vector<Vector<float>> input; 
	std::vector<Vector<float>> output;

	std::vector<Matrix<float>> dE_dw;
	std::vector<Vector<float>> dE_db;

	std::vector<Vector<float>> dE_do;
	std::vector<Vector<float>> dE_dz;

	
	std::vector<Vector<float>> dz_db;
	std::vector<Vector<float>> dz_dw;
	
	std::vector<Vector<float>> feedback;
	
	std::shared_ptr<Layer> preLayer = nullptr, postLayer=nullptr;
	
public:
	DenseLayer();
	DenseLayer(const DenseLayer& rhs);
	DenseLayer(const int& dim);
	DenseLayer(const int& dim, std::string _activation);

	virtual ~DenseLayer();

	virtual void feed_forward(const int& idx);
	virtual void back_propagation(const int& idx);
	virtual void predict();
	virtual void update();
	virtual void init(int batch_size,std::unique_ptr<Optimizer>& _optimizer);
	virtual void setInput(const Vector<float>& _input,const int& idx);
	virtual void setError(const Vector<float>& error,const int& idx);
	virtual void connect(std::shared_ptr<Layer>& cur_layer, std::shared_ptr<Layer>& new_layer);
	virtual void setOptimizer(std::unique_ptr<Optimizer>& _optimizer);
	
	virtual std::shared_ptr<Layer> getPostLayer();
	virtual std::shared_ptr<Layer> getPreLayer();
	virtual Vector<float> getOutput(const int& idx);
	virtual Vector<float> getFeedback(const int& idx);
	virtual int getDim() const; 

	virtual void setActivation(std::string name);
	std::string getActivationName() const;



};

#endif
