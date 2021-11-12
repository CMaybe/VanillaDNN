#ifndef VANILLA_DNN_ACTIVATON_CPP
#define VANILLA_DNN_ACTIVATON_CPP

#include <VanillaDNN/Functions/Activation.hpp>



// Sigmoid
Vector<float> Sigmoid::getActivated(Vector<float>& input){
	Vector<float> output(input.get_size(), 0);
	input = input.clip(-88.72, 88.72);
	for (int i = 0; i < output.get_size(); i++) {
		output(i) = 1 / (1 + exp(-input(i)));
	}
	return output;
}

Matrix<float> Sigmoid::getActivated(Matrix<float>& input){
	Matrix<float> output(input.get_rows_size(), input.get_cols_size(), 0);
	input = input.clip(-88.72, 88.72);
	for (int i = 0; i < output.get_rows_size(); i++) {
		for(int j = 0; j < output.get_cols_size(); j++){
			output(i, j) = 1 / (1 + exp(-input(i, j)));
		}
	}
	return output;
	
}
Vector<float> Sigmoid::getActivatedDiff(Vector<float>& input){
	Vector<float> output(input.get_size(), 0);
	input = input.clip(-88.72, 88.72);
	
	Vector<float> temp = this->getActivated(input);
	for (int i = 0; i < input.get_size(); i++) {
		output(i) = temp(i) * (1 - temp(i));
	}
	
	return output;
}

Matrix<float> Sigmoid::getActivatedDiff(Matrix<float>& input){
	Matrix<float> output(input.get_rows_size(), input.get_cols_size(), 0);
	input = input.clip(-88.72, 88.72);
	
	Matrix<float> temp = this->getActivated(input);
	
	for (int i = 0; i < output.get_rows_size(); i++) {
		for(int j = 0; j < output.get_cols_size(); j++){
			output(i, j) = temp(i, j) * (1 - temp(i, j));
		}
	}
	
	return output;
}


// HyperTan
Vector<float> HyperTan::getActivated(Vector<float>& input){
	Vector<float> output(input.get_size(), 0);
	for (int i = 0; i < output.get_size(); i++) {
		output(i) = tanh(input(i));
	}
	return output;
}

Matrix<float> HyperTan::getActivated(Matrix<float>& input){
	Matrix<float> output(input.get_rows_size(), input.get_cols_size(), 0);
	for (int i = 0; i < output.get_rows_size(); i++) {
		for(int j = 0; j < output.get_cols_size(); j++){
			output(i, j) = tanh(input(i, j));
		}
	}
	return output;
	
}

Vector<float> HyperTan::getActivatedDiff(Vector<float>& input){
	Vector<float> output(input.get_size(), 0);
	Vector<float> temp = this->getActivated(input);
	for (int i = 0; i < input.get_size(); i++) {
		output(i) = 1 - (temp(i) * temp(i));
	}
	return output;
}

Matrix<float> HyperTan::getActivatedDiff(Matrix<float>& input){
	Matrix<float> output(input.get_rows_size(), input.get_cols_size(), 0);
	Matrix<float> temp = this->getActivated(input);
	
	for (int i = 0; i < output.get_rows_size(); i++) {
		for(int j = 0; j < output.get_cols_size(); j++){
			output(i, j) = 1 - (temp(i, j) * temp(i, j));
		}
	}
	return output;
}

// ReLU
Vector<float> ReLU::getActivated(Vector<float>& input){
	Vector<float> output(input.get_size(), 0);
	for (int i = 0; i < output.get_size(); i++) {
		output(i) = std::max(input(i), 0.0f);
	}
	return output;
}

Matrix<float> ReLU::getActivated(Matrix<float>& input){
	Matrix<float> output(input.get_rows_size(), input.get_cols_size(), 0);
	for (int i = 0; i < output.get_rows_size(); i++) {
		for(int j = 0; j < output.get_cols_size(); j++){
			output(i, j) = std::max(input(i, j), 0.0f);
		}
	}
	return output;
	
}

Vector<float> ReLU::getActivatedDiff(Vector<float>& input){
	Vector<float> output(input.get_size(), 0);
	for (int i = 0; i < output.get_size(); i++) {
		output(i) = static_cast<float>(0.0f < input(i));
	}
	return output;
}

Matrix<float> ReLU::getActivatedDiff(Matrix<float>& input){
	Matrix<float> output(input.get_rows_size(), input.get_cols_size(), 0);
	for (int i = 0; i < output.get_rows_size(); i++) {
		for(int j = 0; j < output.get_cols_size(); j++){
			output(i,j) = static_cast<float>(0.0f < input(i,j));
		}
	}
	return output;
}

// LeakyReLU
Vector<float> LeakyReLU::getActivated(Vector<float>& input){
	Vector<float> output(input.get_size(), 0);
	for (int i = 0; i < output.get_size(); i++) {
		output(i) = 0 > input(i) ? 0.01 * input(i) : input(i);
	}
	return output;
}

Matrix<float> LeakyReLU::getActivated(Matrix<float>& input){
	Matrix<float> output(input.get_rows_size(), input.get_cols_size(), 0);
	for (int i = 0; i < output.get_rows_size(); i++) {
		for(int j = 0; j < output.get_cols_size(); j++){
			output(i, j) =  0 > input(i, j) ? 0.01 * input(i, j) : input(i, j);
		}
	}
	return output;
	
}

Vector<float> LeakyReLU::getActivatedDiff(Vector<float>& input){
	Vector<float> output(input.get_size(), 0);
	for (int i = 0; i < output.get_size(); i++) {
		output(i) = 0 > input(i) ? 0.01 : 1;
	}
	return output;
}

Matrix<float> LeakyReLU::getActivatedDiff(Matrix<float>& input){
	Matrix<float> output(input.get_rows_size(), input.get_cols_size(), 0);
	for (int i = 0; i < output.get_rows_size(); i++) {
		for(int j = 0; j < output.get_cols_size(); j++){
			output(i, j) =  0 > input(i, j) ? 0.01 : 1;
		}
	}
	return output;
}


// SoftMax
Vector<float> SoftMax::getActivated(Vector<float>& input){
	Vector<float> output(input.get_size(), 0);
	input = input.clip(-88.72, 88.72);
	float sum = 0;
	for (int i = 0; i < output.get_size(); i++) {
		output(i) = exp(input(i));
		sum += output(i);
	}
	return output / sum;
	
}

Matrix<float> SoftMax::getActivatedDiff2(Vector<float>& input){
	Vector<float> temp = this->getActivated(input);
	Matrix<float> output(temp.get_size(), temp.get_size(), 0);
	for (int i = 0; i < output.get_rows_size(); i++) {
		for(int j = 0; j < output.get_cols_size(); j++){
			output(i, j) =  temp(i) * (static_cast<float>(i==j) - temp[j]);
		}
	}
	return output;
}



#endif