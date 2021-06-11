#ifndef VANILLA_DNN_ACTIVATON_CPP
#define VANILLA_DNN_ACTIVATON_CPP

#include <VanillaDNN/Functions/Activation.hpp>

Vector<float> ACTIVATION_FUNCTION::sigmoid(Vector<float>& input) {
	Vector<float> output(input.get_size(), 0);
	for (int i = 0; i < output.get_size(); i++) {
		output(i) = 1 / (1 + exp(-input(i)));
	}
	return output;
}

Vector<float> ACTIVATION_FUNCTION::hyper_tan(Vector<float>& input) {
	Vector<float> output(input.get_size(), 0);
	for (int i = 0; i < output.get_size(); i++) {
		output(i) = tanh(input(i));
	}
	return output;
}

Vector<float> ACTIVATION_FUNCTION::ReLU(Vector<float>& input) {
	Vector<float> output(input.get_size(), 0);
	for (int i = 0; i < output.get_size(); i++) {
		output(i) = std::max(output(i), 0.0f);
	}
	return output;
}

Vector<float> ACTIVATION_FUNCTION::leaky_ReLU(Vector<float>& input) {
	Vector<float> output(input.get_size(), 0);
	for (int i = 0; i < output.get_size(); i++) {
		output(i) = 0 > input(i) ? 0.01 * input(i) : input(i);
	}
	return output;
}

Vector<float> ACTIVATION_FUNCTION::soft_max(Vector<float>& input) {
	Vector<float> output(input.get_size(), 0);
	float sum = 0;
	for (int i = 0; i < output.get_size(); i++) {
		output(i) = exp(input(i));
		sum += output(i);
	}
	return output / sum;
}



//Derivative of a function
Vector<float> DIFF_FUNCTION::sigmoid_diff(Vector<float>& input) {
	Vector<float> output(input.get_size(), 0);
	Vector<float> temp = ACTIVATION_FUNCTION::sigmoid(input);
	for (int i = 0; i < input.get_size(); i++) {
		output(i) = temp(i) * (1 - temp(i));
	}
	return output;
}

Vector<float> DIFF_FUNCTION::hyper_tan_diff(Vector<float>& input) {
	Vector<float> output(input.get_size(), 0);
	Vector<float> temp = ACTIVATION_FUNCTION::hyper_tan(input);
	for (int i = 0; i < input.get_size(); i++) {
		output(i) = 1 - (temp(i) * temp(i));
	}
	return output;
}

Vector<float> DIFF_FUNCTION::ReLU_diff(Vector<float>& input) {
	Vector<float> output(input.get_size(), 0);
	for (int i = 0; i < output.get_size(); i++) {
		output(i) = static_cast<int>(0.0f < input(i));
	}
	return output;
}

Vector<float> DIFF_FUNCTION::leaky_ReLU_diff(Vector<float>& input) {
	Vector<float> output(input.get_size(), 0);
	for (int i = 0; i < output.get_size(); i++) {
		output(i) = 0 > input(i) ? 0.01 : 1;
	}
	return output;
}


Vector<float> DIFF_FUNCTION::soft_max_diff(Vector<float>& input){
	Vector<float> output(input.get_size(), 0);
	Vector<float> temp = ACTIVATION_FUNCTION::soft_max(input);
	for (int i = 0; i < input.get_size(); i++) {
		for(int j = 0; j < input.get_size(); j++){
			output(i) += temp(i)*((static_cast<int>(i==j)) - temp(j));
		}
	}
	return output;
}


#endif