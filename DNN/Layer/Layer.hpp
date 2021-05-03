#pragma once

#include<functional>
#include<cmath>

using namespace std;

class Layer{
	public:
	Layer();
	virtual ~Chiled(){}
	
	private:
	
	function<float(float)> func;
	func Activation;
	
	public:
	void setActivation(func f);
	
	
}


