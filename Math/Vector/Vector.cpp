#pragma once

#include "Vector.h"

template<typename T>
Vector<T>::Vector(int _rows,int _cols, const T& _init){
	vector.resize(_size,_init);
	size = _size;
}

template<typename T>
Vector<T>::Vector(const Vector<T>& rhs){
	Vector = rhs.vector;
	size = rhs.get_size();
}

template<typename T>
Vector<T>::~Vector(){}

template<typename T>
Vector<T>& Vector<T>::operator=(const Vector<T>& rhs){
	if(&rhs == this) return *this;
	
	int new_size = rhs.get_size();
	
	vector.resize(new_size);
	for(int i =0;i<vector.size();i++){
		Vector[i].resize(new_cols);
	}
	
	for(int i = 0;i<new_size;i++){
		Vector[i]=rhs(i);
	}
	
	size = new_size;
	
	return *this;
}

template<typename T>
Vector<T> Vector<T>::operator+(const Vector<T>& rhs){
	int size = rhs.get_size();
	Vector<T> result(size,0);
	if(this->size!=size)
		throw std::out_of_range("Index out of bounds");

	for(int i = 0;i<size;i++){
		result(i) = this->vector(i)+rhs(i);
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
		result(i) = this->vector(i)-rhs(i);
	}
	return result;
}


template<typename T>
Vector<T>& Vector<T>::operator+=(const Vector<T>& rhs){
	int size = rhs.get_size();
	if(this->size!=size)
		throw std::out_of_range("Index out of bounds");

	for(int i = 0;i<size;i++){
		this->vector(i)+=rhs(i);
	}
	return *this;
}

template<typename T>
Vector<T>& Vector<T>::operator-=(const Vector<T>& rhs){
	int size = rhs.get_size();
	if(this->size!=size)
		throw std::out_of_range("Index out of bounds");

	for(int i = 0;i<size;i++){
		this->vector(i)-=rhs(i);
	}
	return *this;
}



template<typename T>
Matrix<T> Vector<T>::transpose(){
	int row = 1;
	int cols = this->size;
	matrix<T> result(rows,rows,0);
	
	for(int i = 0;i<cols;i++){
		result(0,i) = this->vector(i);
	}
	return result;
}

//scalar


template<typename T>
Vector<T> Vector<T>::operator*(const T& rhs){
	int size = this->size;
	Vector<T> result(size,0);
	
	for(int i = 0;i<rows;i++){
		result(i) = this->vector[i] * rhs;
	}
	
	return result;
}

template<typename T>
Vector<T> Vector<T>::operator/(const T& rhs){
	int size = this->size;
	Vector<T> result(size,0);
	
	for(int i = 0;i<rows;i++){
		result(i) = this->vector[i] / rhs;
	}
	
	return result;
}


//ref
template<typename T>
Vector<T>& Vector<T>::operator*=(const T& rhs){
	int size = this->size;
	Vector<T> result(size,0);
	
	for(int i = 0;i<rows;i++){
		result(i) = this->vector[i] * rhs;
	}
	
	return *this;
}

template<typename T>
Vector<T>& Vector<T>::operator/=(const T& rhs){
	int size = this->size;
	Vector<T> result(size,0);
	
	for(int i = 0;i<rows;i++){
		result(i) = this->vector[i] / rhs;
	}
	
	return *this;
}



template<typename T>
T& Vector<T>::operator()(const int& idx){
	return this->vector[idx];
}

template<typename T>
const T& Vector<T>::operator()(const int& idx) const {
	return this->vector[idx];
}

template<typename T>
int Vector<T>::get_rows_size() const{
	return this->rows;
}

template<typename T>
int Vector<T>::get_cols_size() const{
	return this->cols;
}