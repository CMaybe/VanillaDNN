#ifndef VANILLA_DNN_FUNCTIONS_HPP
#define VANILLA_DNN_FUNCTIONS_HPP


#include"Math/Vector/Vector.hpp"
#include<cmath>

namespace ACTIVATION_FUNCTION{
	Vector<float> sigmoid(Vector<float> input){
		Vector<float> output(input.get_size(),0);
		for(int i = 0;i<output.get_size();i++){
			output(i) = 1/(1+exp(-input(i)));
		}
		return output;
	}
	
	Vector<float> hyper_tan(Vector<float> input){
		Vector<float> output(input.get_size(),0);
		for(int i = 0;i<output.get_size();i++){
			output(i) = tanh(input(i));
		}
		return output;
	}
	
	Vector<float> ReLU(Vector<float> input){
		Vector<float> output(input.get_size(),0);
		for(int i = 0;i<output.get_size();i++){
			output(i) = 0>input(i) ? 0 : input(i);
		}
		return output;
	}
	
	Vector<float> leaky_ReLU(Vector<float> input){
		Vector<float> output(input.get_size(),0);
		for(int i = 0;i<output.get_size();i++){
			output(i) = 0>input(i) ? 0.01*input(i) : input(i);
		}
		return output;
	}
	
}


// y : ans , y_hat : output 
namespace LOSS_FUCTION{
	float mean_squared_error(Vector<float> y_hat, Vector<float> y){
		float ans = 0;
		int N = y.get_size();
		for(int i = 0;i<N;i++){
			ans+= (y_hat(i)-y(i))*(y_hat(i)-y(i));
		}
		return ans/N;
	}
	
	float root_mean_squared_error(Vector<float> y_hat, Vector<float> y){
		float ans = 0;
		int N = y.get_size();
		for(int i = 0;i<N;i++){
			ans+= (y_hat(i)-y(i))*(y_hat(i)-y(i));
		}
		return (float)sqrt(ans/N);
	}
	
	float cross_entropy_error(Vector<float> y_hat, Vector<float> y){
		float ans = 0;
		float delta = 1e-7;
		int N = y.get_size();
		for(int i = 0;i<N;i++){
			ans+= -y(i)*log2(y_hat(i)+delta);
		}
		return ans;
	}
	
	float binary_cross_entropy(Vector<float> t, Vector<float> y){
		float ans = 0;
		int N = y.get_size();
		for(int i = 0;i<N;i++){
			ans+= -y(i)*log(t(i)) - (1-y(i))*log(1-t(i));
		}
		return ans/N;
	}
	
	/*
	float categorical_cross_entropy(Vector<float> t ,Vector<float> y, int c){
		float ans = 0;
		int N = y.get_size();
	}
	*/
}



#endif