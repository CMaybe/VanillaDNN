#ifndef VANILLA_DNN_VECTOR_CPP
#define VANILLA_DNN_VECTOR_CPP

#include "Math/Vector/Vector.hpp"

template<typename T>
Vector<T>::Vector(){
	this->size = 0;
}

template<typename T>
Vector<T>::Vector(int _size, const T& _init){
	this->vector.resize(_size,_init);
	this->size = _size;
}

template<typename T>
Vector<T>::Vector(const Vector<T>& rhs){
	this->vector = rhs.vector;
	this->size = rhs.get_size();
}

template<typename T>
Vector<T>::Vector(const std::vector<T>& rhs){
	this->vector.resize(rhs.size());
	this->size = rhs.size();
	for(int i = 0; i < this->size; i++){
		this->vector[i] = rhs[i];
	}
}

template<typename T>
Vector<T>::~Vector(){}

template<typename T>
void Vector<T>::resize(const int& _size, const T& _init){
	this->vector.resize(_size,_init);
	this->size = _size;
}

template<typename T>
void Vector<T>::resize(const int& _size){
	this->vector.resize(_size,0);
	this->size = _size;
}

template<typename T>
Vector<T>& Vector<T>::operator=(const Vector<T>& rhs){
	if(&rhs == this) return *this;
	
	int new_size = rhs.get_size();
	
	this->vector.resize(new_size);
	
	for(int i = 0;i<new_size;i++){
		this->vector[i]=rhs(i);
	}
	
	this->size = new_size;
	
	return *this;
}

template<typename T>
Vector<T> Vector<T>::operator+(const Vector<T>& rhs){
	int size = rhs.get_size();
	Vector<T> result(size,0);
	if(this->size!=size)
		throw std::out_of_range("Index out of bounds");

	for(int i = 0;i<size;i++){
		result(i) = this->vector[i]+rhs(i);
	}
	return result;
}

template<typename T>
Vector<T> Vector<T>::operator-(const Vector<T>& rhs){
	int size = rhs.get_size();
	Vector<T> result(size,0);
	if(this->size!=size)
		throw std::out_of_range("Index out of bounds");

	for(int i = 0;i<size;i++){
		result(i) = this->vector[i]-rhs(i);
	}
	return result;
}


template<typename T>
Vector<T>& Vector<T>::operator+=(const Vector<T>& rhs){
	int size = rhs.get_size();
	if(this->size!=size)
		throw std::out_of_range("Index out of bounds");

	for(int i = 0;i<size;i++){
		this->vector[i]+=rhs(i);
	}
	return *this;
}

template<typename T>
Vector<T>& Vector<T>::operator-=(const Vector<T>& rhs){
	int size = rhs.get_size();
	if(this->size!=size)
		throw std::out_of_range("Index out of bounds");

	for(int i = 0;i<size;i++){
		this->vector[i]-=rhs(i);
	}
	return *this;
}

//scalar
template<typename T>
Vector<T> Vector<T>::operator*(const T& rhs){
	int size = this->size;
	Vector<T> result(size,0);
	
	for(int i = 0;i<size;i++){
		result(i) = this->vector[i] * rhs;
	}
	
	return result;
}

template<typename T>
Vector<T> Vector<T>::operator/(const T& rhs){
	int size = this->size;
	Vector<T> result(size,0);
	
	for(int i = 0;i<size;i++){
		result(i) = this->vector[i] / rhs;
	}
	
	return result;
}


//ref
template<typename T>
Vector<T>& Vector<T>::operator*=(const T& rhs){
	int size = this->size;
	Vector<T> result(size,0);
	
	for(int i = 0;i<size;i++){
		result(i) = this->vector[i] * rhs;
	}
	
	return *this;
}

template<typename T>
Vector<T>& Vector<T>::operator/=(const T& rhs){
	int size = this->size;
	Vector<T> result(size,0);
	
	for(int i = 0;i<size;i++){
		result(i) = this->vector[i] / rhs;
	}
	
	return *this;
}

template<typename T>
Matrix<T> Vector<T>::transpose(){
	int rows = 1;
	int cols = this->size;
	Matrix<T> result(rows,rows,0);
	
	for(int i = 0;i<cols;i++){
		result(0,i) = this->vector(i);
	}
	return result;
}

template<typename T>
T Vector<T>::dot(const Vector<T>& rhs){
	T result = 0;
	if(this->size!=rhs.get_size())
		throw std::out_of_range("Index out of bounds");
	for(int i =0;i<this->size;i++){
		result += (this->vector(i) * rhs(i));
	}
}

template<typename T>
T& Vector<T>::operator()(const int& idx){
	return this->vector[idx];
}

template<typename T>
T& Vector<T>::operator[](const int& idx){
	return this->vector[idx];
}


template<typename T>
const T& Vector<T>::operator()(const int& idx) const {
	return this->vector[idx];
}

template<typename T>
const T& Vector<T>::operator[](const int& idx) const {
	return this->vector[idx];
}


template<typename T>
int Vector<T>::get_size() const{
	return this->size;
}

#endif