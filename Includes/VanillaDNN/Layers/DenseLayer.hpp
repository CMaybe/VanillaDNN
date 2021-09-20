#ifndef VANILLA_DNN_DENSE_LAYER_HPP
#define VANILLA_DNN_DENSE_LAYER_HPP

#include <vector>
#include <functional>
#include <memory>
#include <VanillaDNN/Layers/Layer.hpp>
#include <VanillaDNN/Math/Matrix/Matrix.hpp>
#include <VanillaDNN/Math/Vector/Vector.hpp>
#include <VanillaDNN/Functions/Functions.hpp>



class DenseLayer : public Layer{
protected:
	using Activation = std::function<Vector<float>(Vector<float>&)>;
	int nNueron;

public:
	DenseLayer();
	DenseLayer(int _nNueron);
	DenseLayer(int _nNueron, Activation _activation);

	Activation activation, activation_diff;


	virtual ~DenseLayer();
	virtual void feed_forward(int idx);
	virtual void back_propagation(int idx);
	virtual void update(const Matrix<float>& dw,const Vector<float>& db);
	
	int getNueronCnt();



};

#endif
