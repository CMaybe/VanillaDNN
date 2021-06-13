#ifndef VANILLA_DNN_OUTPUT_LAYER_HPP
#define VANILLA_DNN_OUTPUT_LAYER_HPP

#include <vector>
#include <functional>
#include <memory>
#include <VanillaDNN/Layers/Layer.hpp>

class OutputLayer : public Layer {
public:
	OutputLayer(int _nNueron);
	virtual ~OutputLayer();

};

#endif
