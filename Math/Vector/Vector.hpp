#ifndef VANILLA_DNN_VECTOR_HPP
#define VANILLA_DNN_VECTOR_HPP

#include<vector>
#include<stdexcept>
#include<iostream>
#include<vector>
#include<cmath>

template<typename T> class Matrix;

template<typename T> class Vector {
private:
	std::vector<T> vector;
	int size;

public:
	Vector();
	Vector(int _size, const T& _init);
	Vector(const Vector<T>& rhs);
	Vector(const std::vector<T>& rhs);
	Vector<T>& operator=(const Vector<T>& rhs);


	virtual ~Vector();



	//operator
	Vector<T> operator+(const Vector<T>& rhs);
	Vector<T> operator-(const Vector<T>& rhs);
	Vector<T> operator*(const Vector<T>& rhs);
	Vector<T>& operator+=(const Vector<T>& rhs);
	Vector<T>& operator-=(const Vector<T>& rhs);
	Vector<T>& operator*=(const Vector<T>& rhs);

	//scalar	
	Vector<T> operator*(const T& rhs);
	Vector<T> operator/(const T& rhs);
	Vector<T>& operator*=(const T& rhs);
	Vector<T>& operator/=(const T& rhs);

	//vector
	Matrix<T> transpose();
	T dot(const Vector<T>& rhs);

	T& operator()(const int& idx);
	const T& operator()(const int& idx) const;
	T& operator()(const int& begin, const int& end);

	T& operator[](const int& idx);
	const T& operator[](const int& idx) const;

	//cout overloading 
	template<typename U>
	friend std::ostream& operator << (std::ostream& out, const Vector<U>& v);

	int get_size() const;
	void push_back(T value);
	void setRandom();
	void resize(const int& _size);
	void resize(const int& _size, const T& _init);
	float norm();

};

template<typename U>
std::ostream& operator << (std::ostream& out, const Vector<U>& v)
{
	int size = v.get_size();
	out << "size : " << size << '\n';
	for (int i = 0; i < size; i++) {
		out << v(i) << '\t';
	}
	out << '\n';
	return out;
}


#include"Vector.cpp"

#endif