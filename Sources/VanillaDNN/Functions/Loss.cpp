#ifndef VANILLA_DNN_LOSS_CPP
#define VANILLA_DNN_LOSS_CPP

#include <VanillaDNN/Functions/Loss.hpp>

// y : ans , y_hat : output 
float LOSS_FUNCTION::mean_squared_error(Vector<float>& y_hat, Vector<float>& y) {
	float ans = 0;
	int N = y.get_size();
	for (int i = 0; i < N; i++) {
		ans += (y_hat(i) - y(i)) * (y_hat(i) - y(i));
	}
	return ans / 2;
}

float LOSS_FUNCTION::root_mean_squared_error(Vector<float>& y_hat, Vector<float>& y) {
	float ans = 0;
	int N = y.get_size();
	for (int i = 0; i < N; i++) {
		ans += (y_hat(i) - y(i)) * (y_hat(i) - y(i));
	}
	return (float)sqrt(ans / 2);
}

float LOSS_FUNCTION::cross_entropy(Vector<float>& y_hat, Vector<float>& y) {
	float ans = 0;
	float delta = 1e-6;
	int N = y.get_size();
	for (int i = 0; i < N; i++) {
		ans += -y(i) * log(y_hat(i) + delta);
	}
	return ans / N;
}

float LOSS_FUNCTION::binary_cross_entropy(Vector<float>& y_hat, Vector<float>& y) {
	float ans = 0;
	float delta = 1e-4;
	int N = y.get_size();
	for (int i = 0; i < N; i++) {
		ans += -y(i) * log(y_hat(i) + delta) - (1 - y(i)) * log(1 - y_hat(i) + delta);
	}
	return ans / N;
}

float LOSS_FUNCTION::categorical_cross_entropy(Vector<float>& y_hat, Vector<float>& y) {
	float ans = 0;
	float delta = 1e-6;
	int N = y.get_size();
	for (int i = 0; i < N; i++) {
		ans += -y(i) * log(y_hat(i) + delta);
	}
	return ans / N;
}

Vector<float> DIFF_FUNCTION::mean_squared_error_diff(Vector<float>& y_hat, Vector<float>& y) {
	int N = y.get_size();
	Vector<float> result(N, 0);
	for (int i = 0; i < N; i++) {
		result(i) = (y_hat(i) - y(i)) / N * 2;
	}
	return result;
}

Vector<float> DIFF_FUNCTION::root_mean_squared_error_diff(Vector<float>& y_hat, Vector<float>& y) {
	int N = y.get_size();
	Vector<float> result(N, 0);
	Vector<float> temp = mean_squared_error_diff(y_hat, y);
	for (int i = 0; i < N; i++) {
		float ori = (y_hat(i) - y(i)) * (y_hat(i) - y(i)) / N;
		result(i) = temp(i) / ori / 2;
	}
	return result;
}

Vector<float> DIFF_FUNCTION::cross_entropy_diff(Vector<float>& y_hat, Vector<float>& y){
	int N = y.get_size();
	Vector<float> result(N, 0);
	float delta = 1e-4;
	for (int i = 0; i < N; i++) {
		result(i) = -y(i) / (y_hat(i) + delta);
	}
	return result / N;
}

Vector<float> DIFF_FUNCTION::binary_cross_entropy_diff(Vector<float>& y_hat, Vector<float>& y) {
	int N = y.get_size();
	Vector<float> result(N, 0);
	for (int i = 0; i < N; i++) {
		result(i) = -y(i) / y_hat(i) + (1 - y(i)) / (1 - y_hat(i));
	}
	return result;
}

Vector<float> DIFF_FUNCTION::categorical_cross_entropy_diff(Vector<float>& y_hat, Vector<float>& y){
	int N = y.get_size();
	Vector<float> result(N, 0);
	result = y_hat - y;
	return result / N;
}

#endif