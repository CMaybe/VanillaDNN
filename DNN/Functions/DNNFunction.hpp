#ifndef VANILLA_DNN_FUNCTIONS_HPP
#define VANILLA_DNN_FUNCTIONS_HPP


#include"Math/Vector/Vector.hpp"
#include<cmath>

namespace ACTIVATION_FUNCTION{
	float sigmoid(float input){
		return 1/(1+exp(-input));
	}
	
	float hyper_tan(float input){
		return tanh(input);
	}
	
	float ReLU(float input){
		return 0>input ? 0 : input;
	}
	
	float leaky_ReLU(float){
		return 0>input ? 0.01*input :input;
	}
	
}


// y : ans , y_hat : output 
namespace LOSS_FUCTION{
	float mean_squared_error(Vector<float> y_hat, Vector<float> y){
		float ans = 0;
		int N = y.size();
		for(int i = 0;i<N;i++){
			ans+= (y_hat(i)-y(i))*(y_hat(i)-y(i));
		}
		return ans/N:
	}
	
	float root_mean_squared_error(Vector<float> y_hat, Vector<float> y){
		float ans = 0;
		int N = y.size();
		for(int i = 0;i<N;i++){
			ans+= (y_hat(i)-y(i))*(y_hat(i)-y(i));
		}
		return (float)sqrt(ans/N):
	}
	
	float cross_entropy_error(Vector<float> y_hat, Vector<float> y){
		float ans = 0;
		float delta = 1e-7
		int N = y.size();
		for(int i = 0;i<N;i++){
			ans+= -y(i)*log2(y_hat(i)+delta);
		}
		return ans:
	}
	
	float binary_cross_entropy(Vector<float> t, Vector<float> y){
		float ans = 0;
		int N = y.size();
		for(int i = 0;i<N;i++){
			ans+= -y(i)*log(t(i)) - (1-y(i))*log(1-t(i));
		}
		return ans/N:
	}
	
	float categorical_cross_entropy(Vector<float> t ,Vector<float> y, int c){
		
	}
}



#endif