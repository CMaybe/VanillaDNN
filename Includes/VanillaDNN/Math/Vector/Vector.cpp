#ifndef VANILLA_DNN_VECTOR_CPP
#define VANILLA_DNN_VECTOR_CPP

#include <VanillaDNN/Math/Vector/Vector.hpp>

template<typename T>
Vector<T>::Vector() {
	this->size = 0;
}

template<typename T>
Vector<T>::Vector(int _size, const T& _init) {
	this->vector.resize(_size, _init);
	this->size = _size;
}

template<typename T>
Vector<T>::Vector(const Vector<T>& rhs) {
	this->vector = rhs.vector;
	this->size = rhs.get_size();
}

template<typename T>
Vector<T>::Vector(const Matrix<T>& rhs) {
	if (this->rhs.get_rows_size() != 1)
		throw std::out_of_range("Index out of bounds");
	this->size = rhs.get_rows_size();
	this->vector.resize(size,0);
	for(int i = 0; i < this->size; i++){
		this->vector[i] = rhs(i,0);
	}
}

template<typename T>
Vector<T>::Vector(const std::vector<T>& rhs) {
	this->vector.resize(rhs.size());
	this->size = rhs.size();
	for (int i = 0; i < this->size; i++) {
		this->vector[i] = rhs[i];
	}
}

template<typename T>
Vector<T>::~Vector() {}

template<typename T>
void Vector<T>::resize(const int& _size, const T& _init) {
	this->vector.resize(_size, _init);
	this->size = _size;
}

template<typename T>
void Vector<T>::resize(const int& _size) {
	this->vector.resize(_size, 0);
	this->size = _size;
}

template<typename T>
Vector<T>& Vector<T>::operator=(const Vector<T>& rhs) {
	if (&rhs == this) return *this;

	this->size = rhs.get_size();
	this->vector.resize(this->size);

	for (int i = 0; i < this->size; i++) {
		this->vector[i] = rhs(i);
	}


	return *this;
}

template<typename T>
Vector<T>& Vector<T>::operator=(const Matrix<T>& rhs) {
	this->size = rhs.get_rows_size();
	this->vector.resize(this->size);

	for (int i = 0; i < this->size; i++) {
		this->vector[i] = rhs(i,0);
	}

	return *this;
}

template<typename T>
Vector<T> Vector<T>::operator+(const Vector<T>& rhs) {
	int size = rhs.get_size();
	Vector<T> result(size, 0);
	if (this->size != size)
		throw std::out_of_range("Index out of bounds");

	for (int i = 0; i < size; i++) {
		result(i) = this->vector[i] + rhs(i);
	}
	return result;
}

template<typename T>
Vector<T> Vector<T>::operator-(const Vector<T>& rhs) {
	int size = rhs.get_size();
	Vector<T> result(size, 0);
	if (this->size != size)
		throw std::out_of_range("Index out of bounds");

	for (int i = 0; i < size; i++) {
		result(i) = this->vector[i] - rhs(i);
	}
	return result;
}

template<typename T>
Vector<T> Vector<T>::operator*(const Vector<T>& rhs) {
	int size = rhs.get_size();
	Vector<T> result(size, 0);
	if (this->size != size)
		throw std::out_of_range("Index out of bounds");

	for (int i = 0; i < size; i++) {
		result(i) = this->vector[i] * rhs(i);
	}
	return result;
}


template<typename T>
Vector<T>& Vector<T>::operator+=(const Vector<T>& rhs) {
	int size = rhs.get_size();
	if (this->size != size)
		throw std::out_of_range("Index out of bounds");

	for (int i = 0; i < size; i++) {
		this->vector[i] += rhs(i);
	}
	return *this;
}

template<typename T>
Vector<T>& Vector<T>::operator-=(const Vector<T>& rhs) {
	int size = rhs.get_size();
	if (this->size != size)
		throw std::out_of_range("Index out of bounds");

	for (int i = 0; i < size; i++) {
		this->vector[i] -= rhs(i);
	}
	return *this;
}

template<typename T>
Vector<T>& Vector<T>::operator*=(const Vector<T>& rhs) {
	int size = rhs.get_size();
	if (this->size != size)
		throw std::out_of_range("Index out of bounds");

	for (int i = 0; i < size; i++) {
		this->vector[i] *= rhs(i);
	}
	return *this;
}

//scalar
template<typename T>
Vector<T> Vector<T>::operator+(const T& rhs) {
	int size = this->size;
	Vector<T> result(size, 0);

	for (int i = 0; i < size; i++) {
		result(i) = this->vector[i] + rhs;
	}

	return result;
}

template<typename T>
Vector<T> Vector<T>::operator-(const T& rhs) {
	int size = this->size;
	Vector<T> result(size, 0);

	for (int i = 0; i < size; i++) {
		result(i) = this->vector[i] - rhs;
	}

	return result;
}

template<typename T>
Vector<T> Vector<T>::operator*(const T& rhs) {
	int size = this->size;
	Vector<T> result(size, 0);

	for (int i = 0; i < size; i++) {
		result(i) = this->vector[i] * rhs;
	}

	return result;
}

template<typename T>
Vector<T> Vector<T>::operator/(const T& rhs) {
	int size = this->size;
	Vector<T> result(size, 0);

	for (int i = 0; i < size; i++) {
		result(i) = this->vector[i] / rhs;
	}

	return result;
}


//ref
template<typename T>
Vector<T>& Vector<T>::operator+=(const T& rhs) {
	int size = this->size;

	for (int i = 0; i < size; i++) {
		 this->vector[i] += rhs;
	}

	return *this;
}

template<typename T>
Vector<T>& Vector<T>::operator-=(const T& rhs) {
	int size = this->size;

	for (int i = 0; i < size; i++) {
		 this->vector[i] -= rhs;
	}

	return *this;
}

template<typename T>
Vector<T>& Vector<T>::operator*=(const T& rhs) {
	int size = this->size;

	for (int i = 0; i < size; i++) {
		 this->vector[i] *= rhs;
	}

	return *this;
}

template<typename T>
Vector<T>& Vector<T>::operator/=(const T& rhs) {
	int size = this->size;

	for (int i = 0; i < size; i++) {
		 this->vector[i] /= rhs;
	}

	return *this;
}

template<typename T>
bool Vector<T>::operator==(const Vector<T>& rhs){
	return (*this - rhs).norm() < 0.001f;
}

template<typename T>
Matrix<T> Vector<T>::transpose(){
	int rows = 1;
	int cols = this->size;
	Matrix<T> result(1, cols, 0);

	for (int i = 0; i < cols; i++) {
		result(0, i) = this->vector[i];
	}
	return result;
}

template<typename T>
T Vector<T>::dot(const Vector<T>& rhs) {
	T result = 0;
	if (this->size != rhs.get_size())
		throw std::out_of_range("Index out of bounds");
	for (int i = 0; i < this->size; i++) {
		result += (this->vector(i) * rhs(i));
	}
	return result;
}

template<typename T>
Matrix<T> Vector<T>::dot(const Matrix<T>& rhs) {
	Matrix<T> result(this->size, rhs.get_cols_size(), 0);
	for (int i = 0; i < this->size; i++) {
		for(int j = 0;j < rhs.get_cols_size();j++){
			result(i,j) = this->vector[i] * rhs(0, j);
		}
	}
	return result;
}

template<typename T>
T& Vector<T>::operator()(const int& idx) {
	return this->vector[idx];
}

template<typename T>
T& Vector<T>::operator[](const int& idx) {
	return this->vector[idx];
}


template<typename T>
const T& Vector<T>::operator()(const int& idx) const {
	return this->vector[idx];
}

template<typename T>
T& Vector<T>::operator()(const int& begin, const int& end)
{
	Vector<T> result = *this;
	result.resize(end - begin);
	result.vector = result.vector(begin, end);
	return result;
}

template<typename T>
const T& Vector<T>::operator[](const int& idx) const {
	return this->vector[idx];
}


template<typename T>
int Vector<T>::get_size() const {
	return this->size;
}


template<typename T>
void Vector<T>::push_back(T value) {
	this->vector.push_back(value);
	this->size = vector.size();
}

template<typename T>
void Vector<T>::setRandom()
{
	std::srand(std::time(0));
	for (int i = 0; i < this->size; i++) {
		this->vector[i] = ((float)std::rand() / ((float)RAND_MAX)) * 2 - 1;
	}
}

template<typename T>
float Vector<T>::norm(){
	float sum = 0;
	for (int i = 0; i < this->size; i++) {
		sum += static_cast<float>(this->vector[i] * this->vector[i]);
	}
	return std::sqrt(sum);
}

template<typename T>
Vector<T> Vector<T>::square() {
	int size = this->size;
	Vector<T> result(size, 0);

	for (int i = 0; i < size; i++) {
		result(i) = this->vector[i] * this->vector[i];
	}
	return result;
}

template<typename T>
Vector<T> Vector<T>::sqrt() {
	int size = this->size;
	Vector<T> result(size, 0);

	for (int i = 0; i < size; i++) {
		result(i) = std::sqrt(this->vector[i]);
	}
	return result;
}


template<typename T>
Vector<T> Vector<T>::inv() {
	int size = this->size;
	Vector<T> result(size, 0);

	for (int i = 0; i < size; i++) {
		result(i) = 1.0f / this->vector[i];
	}
	return result;
}


template<typename T>
Vector<T> Vector<T>::onehot(){
	int size = this->size;
	Vector<T> result(size, 0);
	float _max = *std::max_element(this->vector.begin(), this->vector.end());
	for(int i = 0; i < size; i++){
		if (_max == this->vector[i]){
			result[i] = 1;
			break;
		}
	}
	return result;
}

template<typename T>
Vector<T> Vector<T>::clip(const T& _min, const T& _max){
	int size = this->size;
	Vector<T> result(size, 0);
	for(int i = 0; i < size; i++){
		if (_max < this->vector[i]) result[i] = _max;
		else if (_min > this->vector[i]) result[i] = _min;
		else result[i] = this->vector[i];
	}
	return result;
}

template<typename T>
T Vector<T>::sum(){
	T result = 0;
	for(int i = 0; i < size; i++){
		result += this-vector[i];
	}
	return result;
}

#endif