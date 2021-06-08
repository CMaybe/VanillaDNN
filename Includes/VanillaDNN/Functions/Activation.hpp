#ifndef VANILLA_DNN_ACTIVATON_HPP
#define VANILLA_DNN_ACTIVATON_HPP


#include <VanillaDNN/Math/Vector/Vector.hpp>
#include <cmath>

namespace ACTIVATION_FUNCTION {
	Vector<float> sigmoid(Vector<float> input);
	Vector<float> hyper_tan(Vector<float> input);
	Vector<float> ReLU(Vector<float> input);
	Vector<float> leaky_ReLU(Vector<float> input);
	Vector<float> soft_max(Vector<float> input);
}

//Derivative of a function
namespace DIFF_FUNCTION {
	Vector<float> sigmoid_diff(Vector<float> input);
	Vector<float> hyper_tan_diff(Vector<float> input);
	Vector<float> ReLU_diff(Vector<float> input);
	Vector<float> leaky_ReLU_diff(Vector<float> input);
	Vector<float> soft_max_diff(Vector<float> input);
}



#endif