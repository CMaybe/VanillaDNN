#include "Math/Matrix/Matrix.hpp"
#include "Math/Vector/Vector.hpp"
#include "DNN/Layer/Layer.hpp"
#include "DNN/Functions/DNNFunction.hpp"

#include <iostream>
#include <cstdlib>
using namespace std;

int main(int argc, char **argv) {
	Matrix<float> m1(3, 3, 0);
	Matrix<float> m2(3, 3, 0);
	
	for(int i = 0;i<3;i++){
		for(int j=0;j<3;j++){
			m1(i,j) = rand() % 10;
			m2(i,j) = rand() % 10;
		}
	}
	
	cout<<m1<<m2;
	cout<<m1 + m2;
	cout<<m1 - m2;
	cout<<m1 * m2;
	
	cout<<m1 + 10;
	cout<<m1 - 10;
	cout<<m1 * 10;
	cout<<m1 / 10;
	
	
	
	return 0;
}