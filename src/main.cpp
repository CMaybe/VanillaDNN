#include "Math/Matrix/Matrix.hpp"
#include "Math/Vector/Vector.hpp"
#include <iostream>
#include <cstdlib>
using namespace std;

int main(int argc, char **argv) {
	Matrix<int> m1(3, 3, 0);
	Matrix<int> m2(3, 3, 0);
	Vector<int> v(3,2);
	
	
	
	for(int i = 0;i<3;i++){
		v(i) = rand()%10;
		for(int j=0;j<3;j++){
			m1(i,j) = rand() % 10;
			m2(i,j) = rand() % 10;
		}
	}
	
	cout<<v;
	v=v*10;
	cout<<v;
	cout<<m1 + m2;
	cout<<m1-m2<<endl;
	cout<<m1*m2<<endl;
	cout<<m1.transpose()<<endl;
	
	
	cout<< m1 * v;
	return 0;
}