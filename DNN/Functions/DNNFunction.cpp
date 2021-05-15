#ifndef VANILLA_DNN_FUNCTIONS_CPP
#define VANILLA_DNN_FUNCTIONS_CPP


#include"DNN/Functions/DNNFunction.hpp"

Vector<float> ACTIVATION_FUNCTION::sigmoid(Vector<float> input){
	Vector<float> output(input.get_size(),0);
	for(int i = 0;i<output.get_size();i++){
		output(i) = 1/(1+exp(-input(i)));
	}
	return output;
}

Vector<float> ACTIVATION_FUNCTION::hyper_tan(Vector<float> input){
	Vector<float> output(input.get_size(),0);
	for(int i = 0;i<output.get_size();i++){
		output(i) = tanh(input(i));
	}
	return output;
}

Vector<float> ACTIVATION_FUNCTION::ReLU(Vector<float> input){
	Vector<float> output(input.get_size(),0);
	for(int i = 0;i<output.get_size();i++){
		output(i) = 0>input(i) ? 0 : input(i);
	}
	return output;
}

Vector<float> ACTIVATION_FUNCTION::leaky_ReLU(Vector<float> input){
	Vector<float> output(input.get_size(),0);
	for(int i = 0;i<output.get_size();i++){
		output(i) = 0>input(i) ? 0.01*input(i) : input(i);
	}
	return output;
}

Vector<float> ACTIVATION_FUNCTION::soft_max(Vector<float> input){
	Vector<float> output(input.get_size(),0);
	float sum=0;
	for(int i = 0;i<output.get_size();i++){
		output(i) = exp(input(i));
		sum += output(i);
	}
	return output/sum;
}


// y : ans , y_hat : output 

float LOSS_FUNCTION::mean_squared_error(Vector<float> y_hat, Vector<float> y){
	float ans = 0;
	int N = y.get_size();
	for(int i = 0;i<N;i++){
		ans+= (y_hat(i)-y(i))*(y_hat(i)-y(i));
	}
	return ans/N;
}

float LOSS_FUNCTION::root_mean_squared_error(Vector<float> y_hat, Vector<float> y){
	float ans = 0;
	int N = y.get_size();
	for(int i = 0;i<N;i++){
		ans+= (y_hat(i)-y(i))*(y_hat(i)-y(i));
	}
	return (float)sqrt(ans/N);
}

float LOSS_FUNCTION::cross_entropy_error(Vector<float> y_hat, Vector<float> y){
	float ans = 0;
	float delta = 1e-7;
	int N = y.get_size();
	for(int i = 0;i<N;i++){
		ans+= -y(i)*log(y_hat(i)+delta);
	}
	return ans;
}

float LOSS_FUNCTION::binary_cross_entropy(Vector<float> y_hat, Vector<float> y){
	float ans = 0;
	int N = y.get_size();
	for(int i = 0;i<N;i++){
		ans+= -y(i)*log(y_hat(i)) - (1-y(i))*log(1-y_hat(i));
	}
	return ans/N;
}

/*
float LOSS_FUNCTION::categorical_cross_entropy(Vector<float> t ,Vector<float> y, int c){
	float ans = 0;
	int N = y.get_size();
}
*/

//Derivative of a function
Vector<float> DIFF_FUNCTION::sigmoid_diff(Vector<float> input){
	Vector<float> output(input.get_size(),0);
	Vector<float> temp = ACTIVATION_FUNCTION::sigmoid(input);
	for(int i = 0 ;i < input.get_size();i++){
		output(i) = temp(i)*(1-temp(i));
	}
	return output;
}

Vector<float> DIFF_FUNCTION::hyper_tan_diff(Vector<float> input){
	Vector<float> output(input.get_size(),0);
	Vector<float> temp = ACTIVATION_FUNCTION::hyper_tan(input);
	for(int i = 0 ;i < input.get_size();i++){
		output(i) = 1-(temp(i)*temp(i));
	}
	return output;
}

Vector<float> DIFF_FUNCTION::ReLU_diff(Vector<float> input){
	Vector<float> output(input.get_size(),0);
	for(int i = 0;i<output.get_size();i++){
		output(i) = 0>input(i) ? 0 : 1;
	}
	return output;
}

Vector<float> DIFF_FUNCTION::leaky_ReLU_diff(Vector<float> input){
	Vector<float> output(input.get_size(),0);
	for(int i = 0;i<output.get_size();i++){
		output(i) = 0>input(i) ? 0.01 : 1;
	}
	return output;
}

/*
Vector<float> DIFF_FUNCTION::soft_max_diff(Vector<float> input){
	Vector<float> output(input.get_size(),0);
	float sum=0;
	for(int i = 0;i<output.get_size();i++){
		output(i) = exp(input(i));
		sum += output(i);
	}
	return output/sum;
}
*/

Vector<float> DIFF_FUNCTION::mean_squared_error_diff(Vector<float> y_hat, Vector<float> y){
	int N = y.get_size();
	Vector<float> result(N,0);
	for(int i = 0;i<N;i++){
		result(i) = (y_hat(i)-y(i))/N*2;
	}
	return result;
}

Vector<float> DIFF_FUNCTION::root_mean_squared_error_diff(Vector<float> y_hat, Vector<float> y){
	int N = y.get_size();
	Vector<float> result(N,0);
	Vector<float> temp = mean_squared_error_diff(y_hat,y);
	for(int i = 0;i<N;i++){
		float ori = (y_hat(i)-y(i))*(y_hat(i)-y(i))/N;
		result(i) = temp(i)/ori/2;
	}
	return result;
}

Vector<float> DIFF_FUNCTION::cross_entropy_error_diff(Vector<float> y_hat, Vector<float> y){
	int N = y.get_size();
	Vector<float> result(N,0);
	float delta = 1e-7;
	for(int i = 0;i<N;i++){
		result(i) = -y(i)/(y_hat(i)+delta);
	}
	return result;
}

Vector<float> DIFF_FUNCTION::binary_cross_entropy_diff(Vector<float> y_hat, Vector<float> y){
	int N = y.get_size();
	Vector<float> result(N,0);
	for(int i = 0;i<N;i++){
		result(i) = -y(i)/y_hat(i) + (1-y(i))/(1-y_hat(i));
	}
	return result;
}

/*
float DIFF_FUNCTION::categorical_cross_entropy(Vector<float> t ,Vector<float> y, int c){
	float ans = 0;
	int N = y.get_size();
}
*/



#endif