#ifndef VANILLA_DNN_LOSS_HPP
#define VANILLA_DNN_LOSS_HPP

#include <VanillaDNN/Math/Vector/Vector.hpp>
#include <cmath>

namespace LOSS_FUNCTION {
	float mean_squared_error(Vector<float> y_hat, Vector<float> y);
	float root_mean_squared_error(Vector<float> y_hat, Vector<float> y);
	float categorical_cross_entropy(Vector<float> y_hat, Vector<float> y);
	float binary_cross_entropy(Vector<float> y_hat, Vector<float> y);
}

//Derivative of a function
namespace DIFF_FUNCTION {
	Vector<float> mean_squared_error_diff(Vector<float> y_hat, Vector<float> y);
	Vector<float> root_mean_squared_error_diff(Vector<float> y_hat, Vector<float> y);
	Vector<float> categorical_cross_entropy_diff(Vector<float> y_hat, Vector<float> y);
	Vector<float> binary_cross_entropy_diff(Vector<float> y_hat, Vector<float> y);
}

#endif