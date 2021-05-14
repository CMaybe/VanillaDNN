#ifndef VANILLA_DNN_FUNCTIONS_HPP
#define VANILLA_DNN_FUNCTIONS_HPP


#include"Math/Vector/Vector.hpp"
#include<cmath>

namespace ACTIVATION_FUNCTION{
	Vector<float> sigmoid(Vector<float> input);
	Vector<float> hyper_tan(Vector<float> input);
	Vector<float> ReLU(Vector<float> input);
	Vector<float> leaky_ReLU(Vector<float> input);
	Vector<float> soft_max(Vector<float> input);
}


// y : ans , y_hat : output 
namespace LOSS_FUCTION{
	float mean_squared_error(Vector<float> y_hat, Vector<float> y);	
	float root_mean_squared_error(Vector<float> y_hat, Vector<float> y);
	float cross_entropy_error(Vector<float> y_hat, Vector<float> y);
	float binary_cross_entropy(Vector<float> y_hat, Vector<float> y);
	//float categorical_cross_entropy(Vector<float> t ,Vector<float> y, int c);
}

//Derivative of a function
namespace DIFF_FUNCTION{
	Vector<float> sigmoid_diff(Vector<float> input);
	Vector<float> hyper_tan_diff(Vector<float> input);
	Vector<float> ReLU_diff(Vector<float> input);
	Vector<float> leaky_ReLU_diff(Vector<float> input);
	//Vector<float> soft_max_diff(Vector<float> input);
	Vector<float> mean_squared_error_diff(Vector<float> y_hat, Vector<float> y);
	Vector<float> root_mean_squared_error_diff(Vector<float> y_hat, Vector<float> y);
	Vector<float> cross_entropy_error_diff(Vector<float> y_hat, Vector<float> y);
	Vector<float> binary_cross_entropy_diff(Vector<float> y_hat, Vector<float> y);
	//float categorical_cross_entropy_diff(Vector<float> t ,Vector<float> y, int c);
}



#endif