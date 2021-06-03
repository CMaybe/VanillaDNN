#ifndef VANILLA_DNN_OUTPUT_LAYER_HPP
#define VANILLA_DNN_OUTPUT_LAYER_HPP

#include<vector>
#include<functional>
#include<memory>

#incldue "DNN/Layer/Layer.hpp"
#include "Math/Matrix/Matrix.hpp"
#include "Math/Vector/Vector.hpp"
#include "DNN/Functions/DNNFunction.hpp"



class OutputLayer : public Layer {

public:

	OutputLayer(int _nNueron);
	OutputLayer(int _nNueron, Activation _activation);
	
	virtual ~OutputLayer();
	


};

#endif
