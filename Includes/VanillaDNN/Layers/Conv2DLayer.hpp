#ifndef VANILLA_DNN_INPUT_LAYER_HPP
#define VANILLA_DNN_INPUT_LAYER_HPP

#include <vector>
#include <functional>
#include <memory>
#include <VanillaDNN/Layers/Layer.hpp>

class InputLayer {
private:
	using Activation = std::function<Vector<float>(Vector<float>)>;
	int nNueron;

public:
	InputLayer(int _nNueron);
	std::vector<Vector<float>> inputNeuron; // before activate : preLayer->weight * neuronOutput
	std::vector<Vector<float>> outputNeuron; // after actionte : actinvation(neuronInput)

	virtual ~InputLayer();



};

#endif
