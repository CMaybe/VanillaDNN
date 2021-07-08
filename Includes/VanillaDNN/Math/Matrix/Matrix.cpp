#ifndef VANILLA_DNN_MATRIX_CPP
#define VANILLA_DNN_MATRIX_CPP

#include <VanillaDNN/Math/Matrix/Matrix.hpp>
// #include "Matrix.hpp"

#pragma region MatrixBase
template<typename T,size_t _Rows,size_t _Cols> 
MatrixBase<T, _Rows, _Cols>::MatrixBase() {
	rows = _Rows;
	cols = _Cols;
}

template<typename T,size_t _Rows,size_t _Cols> 
MatrixBase<T, _Rows, _Cols>::MatrixBase(const T& _init) {
	matrix.resize(_Rows);
	for (int i = 0; i < matrix.size(); i++) {
		matrix[i].resize(_Cols, _init);
	}
	rows = _Rows;
	cols = _Cols;
}

template<typename T,size_t _Rows,size_t _Cols>
MatrixBase<T, _Rows, _Cols>::MatrixBase(const MatrixBase<T, _Rows, _Cols>& rhs) {
	this->matrix = rhs.matrix;
	this->rows = rhs.get_rows_size();
	this->cols = rhs.get_cols_size();
}

template<typename T,size_t _Rows,size_t _Cols>
template<size_t Other_Rows,size_t Other_Cols>
MatrixBase<T, _Rows, _Cols>::MatrixBase(const MatrixBase<T, Other_Rows, Other_Cols>& rhs) {
	this->matrix = rhs.matrix;
	this->rows = rhs.get_rows_size();
	this->cols = rhs.get_cols_size();
}

template<typename T,size_t _Rows,size_t _Cols>
MatrixBase<T, _Rows, _Cols>::MatrixBase(const std::vector<std::vector<T>>& rhs) {
	this->rows = rhs.size();
	this->cols = rhs[0].size();
	this->matrix.resize(this->rows);
	for (int i = 0; i < this->rows; i++) {
		this->matrix[i].resize(this->cols, 0);
		for (int j = 0; j < this->cols; j++) {
			this->matrix[i][j] = rhs[i][j];
		}
	}
}

template<typename T,size_t _Rows,size_t _Cols>
MatrixBase<T, _Rows, _Cols>::~MatrixBase() {}


template<typename T,size_t _Rows,size_t _Cols>
void MatrixBase<T, _Rows, _Cols>::resize(int _rows, int _cols, T _init) {
	matrix.resize(_rows);
	for (int i = 0; i < matrix.size(); i++) {
		matrix[i].resize(_cols, _init);
	}
	rows = _rows;
	cols = _cols;
}

template<typename T,size_t _Rows,size_t _Cols> 
void MatrixBase<T, _Rows, _Cols>::setRandom()
{
	std::srand(std::time(0));
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			this->matrix[i][j] = (static_cast<T>(std::rand()) / static_cast<T>(RAND_MAX)) * 2 - 1;
		}
	}
}

template<typename T,size_t _Rows,size_t _Cols>
template<size_t Other_Rows,size_t Other_Cols>
MatrixBase<T, _Rows, _Cols>& MatrixBase<T, _Rows, _Cols>::operator=(const MatrixBase<T, Other_Rows, Other_Cols>& rhs) {
	if (&rhs == this) return *this;

	int new_rows = rhs.get_rows_size();
	int new_cols = rhs.get_cols_size();

	this->matrix.resize(new_rows);
	for (int i = 0; i < matrix.size(); i++) {
		matrix[i].resize(new_cols);
	}

	for (int i = 0; i < new_rows; i++) {
		for (int j = 0; j < new_cols; j++) {
			matrix[i][j] = rhs(i, j);
		}
	}

	rows = new_rows;
	cols = new_cols;

	return *this;
}



template<typename T,size_t _Rows,size_t _Cols> 
template<size_t Other_Rows,size_t Other_Cols>
MatrixBase<T, _Rows, _Cols> MatrixBase<T, _Rows, _Cols>::operator+(const MatrixBase<T, Other_Rows, Other_Cols>& rhs) {
	int rows = rhs.get_rows_size();
	int cols = rhs.get_cols_size();
	MatrixBase<T, _Rows, _Cols> result(rows, cols, 0);
	if (this->cols != cols || this->rows != rows)
		throw std::out_of_range("Index out of bounds");

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result(i, j) = this->matrix[i][j] + rhs(i, j);
		}
	}
	return result;
}

template<typename T,size_t _Rows,size_t _Cols>
template<size_t Other_Rows,size_t Other_Cols>
MatrixBase<T, _Rows, _Cols> MatrixBase<T, _Rows, _Cols>::operator-(const MatrixBase<T, Other_Rows, Other_Cols>& rhs) {
	int rows = rhs.get_rows_size();
	int cols = rhs.get_cols_size();

	MatrixBase<T, _Rows, _Cols> result(rows, cols, 0);
	if (this->cols != cols || this->rows != rows)
		throw std::out_of_range("Index out of bounds");

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result(i, j) = this->matrix[i][j] - rhs(i, j);
		}
	}
	return result;
}

template<typename T,size_t _Rows,size_t _Cols>
template<size_t Other_Rows,size_t Other_Cols>
MatrixBase<T, _Rows, Other_Cols> MatrixBase<T, _Rows, _Cols>::operator*(const MatrixBase<T, Other_Rows, Other_Cols>& rhs) {
	int rows = rhs.get_rows_size();
	int cols = rhs.get_cols_size();
	MatrixBase<T, _Rows, _Cols> result(rows, cols, 0);

	if (this->cols != rows)
		throw std::out_of_range("Index out of bounds");

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			for (int k = 0; k < rows; k++) {
				result(i, j) += this->matrix[i][k] * rhs(k, j);
			}
		}
	}

	return result;
}


template<typename T,size_t _Rows,size_t _Cols>
template<size_t Other_Rows,size_t Other_Cols>
MatrixBase<T, _Rows, _Cols>& MatrixBase<T, _Rows, _Cols>::operator+=(const MatrixBase<T, Other_Rows, Other_Cols>& rhs) {
	int rows = rhs.get_rows_size();
	int cols = rhs.get_cols_size();
	if (this->cols != cols || this->rows != rows)
		throw std::out_of_range("Index out of bounds");

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			this->matrix[i][j] += rhs(i, j);
		}
	}

	return *this;
}

template<typename T,size_t _Rows,size_t _Cols> 
template<size_t Other_Rows,size_t Other_Cols>
MatrixBase<T, _Rows, _Cols>& MatrixBase<T, _Rows, _Cols>::operator-=(const MatrixBase<T, Other_Rows, Other_Cols>& rhs) {
	int rows = rhs.get_rows_size();
	int cols = rhs.get_cols_size();
	if (this->cols != cols || this->rows != rows)
		throw std::out_of_range("Index out of bounds");

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			this->matrix[i][j] -= rhs(i, j);
		}
	}

	return *this;
}


template<typename T,size_t _Rows,size_t _Cols>
template<size_t Other_Rows,size_t Other_Cols>
MatrixBase<T, _Rows, Other_Cols>& MatrixBase<T, _Rows, _Cols>::operator*=(const MatrixBase<T, Other_Rows, Other_Cols>& rhs) {
	MatrixBase<T,_Rows,Other_Cols> result = (*this) * rhs;
	(*this) = result;
	return *this;
}


//scalar
template<typename T,size_t _Rows,size_t _Cols>
MatrixBase<T, _Rows, _Cols> MatrixBase<T, _Rows, _Cols>::operator+(const T& rhs) {
	int rows = this->rows;
	int cols = this->cols;
	MatrixBase<T, _Rows, _Cols> result(rows, cols, 0);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result(i, j) = this->matrix[i][j] + rhs;
		}
	}

	return result;
}

template<typename T,size_t _Rows,size_t _Cols>
MatrixBase<T, _Rows, _Cols> MatrixBase<T, _Rows, _Cols>::operator-(const T& rhs) {
	int row = this->rows;
	int cols = this->cols;
	MatrixBase<T, _Rows, _Cols> result(rows, cols, 0);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result(i, j) = this->matrix[i][j] - rhs;
		}
	}

	return result;
}

template<typename T,size_t _Rows,size_t _Cols>
MatrixBase<T, _Rows, _Cols> MatrixBase<T, _Rows, _Cols>::operator*(const T& rhs) {
	int row = this->rows;
	int cols = this->cols;
	MatrixBase<T, _Rows, _Cols> result(rows, cols, 0);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result(i, j) = this->matrix[i][j] * rhs;
		}
	}

	return result;
}

template<typename T,size_t _Rows,size_t _Cols>
MatrixBase<T, _Rows, _Cols> MatrixBase<T, _Rows, _Cols>::operator/(const T& rhs) {
	int row = this->rows;
	int cols = this->cols;
	MatrixBase<T, _Rows, _Cols> result(rows, cols, 0);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result(i, j) = this->matrix[i][j] / rhs;
		}
	}

	return result;
}


//ref
template<typename T,size_t _Rows,size_t _Cols>
MatrixBase<T, _Rows, _Cols>& MatrixBase<T, _Rows, _Cols>::operator+=(const T& rhs) {
	int row = this->rows;
	int cols = this->cols;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			this->matrix[i][j] += rhs;
		}
	}

	return *this;
}

template<typename T,size_t _Rows,size_t _Cols> 
MatrixBase<T, _Rows, _Cols>& MatrixBase<T, _Rows, _Cols>::operator-=(const T& rhs) {
	int row = this->rows;
	int cols = this->cols;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			this->matrix[i][j] -= rhs;
		}
	}

	return *this;
}

template<typename T,size_t _Rows,size_t _Cols>
MatrixBase<T, _Rows, _Cols>& MatrixBase<T, _Rows, _Cols>::operator*=(const T& rhs) {
	int row = this->rows;
	int cols = this->cols;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			this->matrix[i][j] *= rhs;
		}
	}

	return *this;
}

template<typename T,size_t _Rows,size_t _Cols>
MatrixBase<T, _Rows, _Cols>& MatrixBase<T, _Rows, _Cols>::operator/=(const T& rhs) {
	int row = this->rows;
	int cols = this->cols;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			this->matrix[i][j] /= rhs;
		}
	}

	return *this;
}


template<typename T,size_t _Rows,size_t _Cols>
MatrixBase<T, _Cols, _Rows> MatrixBase<T, _Rows, _Cols>::transpose() {
	int row = this->rows;
	int cols = this->cols;
	MatrixBase<T, _Cols, _Rows> result(0);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result(j, i) = this->matrix[i][j];
		}
	}
	return result;
}

template<typename T,size_t _Rows,size_t _Cols>
MatrixBase<T, _Rows, _Cols> MatrixBase<T, _Rows, _Cols>::square() {
	int rows = this->rows;
	int cols = this->cols;
	MatrixBase<T, _Rows, _Cols> result(rows, cols, 0);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result(i, j) = this->matrix[i][j] * this->matrix[i][j];
		}
	}
	return result;
}

template<typename T,size_t _Rows,size_t _Cols>
MatrixBase<T, _Rows, _Cols> MatrixBase<T, _Rows, _Cols>::inv(){
	int rows = this->rows;
	int cols = this->cols;
	MatrixBase<T, _Rows, _Cols> result(rows, cols, 0);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result(i, j) = 1.0f / this->matrix[i][j];
		}
	}
	return result;
}

template<typename T,size_t _Rows,size_t _Cols>
MatrixBase<T, _Rows, _Cols> MatrixBase<T, _Rows, _Cols>::sqrt(){
	int rows = this->rows;
	int cols = this->cols;
	MatrixBase<T, _Rows, _Cols> result(rows, cols, 0);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result(i, j) = std::sqrt(this->matrix[i][j]);
		}
	}
	return result;
}

template<typename T,size_t _Rows,size_t _Cols>
float MatrixBase<T, _Rows, _Cols>::norm(){
	float sum = 0;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			sum += static_cast<float>(this->matrix[i][j] * this->matrix[i][j]);
		}
	}
	return std::sqrt(sum);
}

template<typename T,size_t _Rows,size_t _Cols>
T& MatrixBase<T, _Rows, _Cols>::operator()(const int& row, const int& col) {
	return this->matrix[row][col];
}

template<typename T,size_t _Rows,size_t _Cols> 
const T& MatrixBase<T, _Rows, _Cols>::operator()(const int& row, const int& col) const {
	return this->matrix[row][col];
}

template<typename T,size_t _Rows,size_t _Cols> 
int MatrixBase<T, _Rows, _Cols>::get_rows_size() const {
	return this->rows;
}

template<typename T,size_t _Rows,size_t _Cols> 
int MatrixBase<T, _Rows, _Cols>::get_cols_size() const {
	return this->cols;
}




// Vector

template<typename T, size_t _Rows> 
MatrixBase<T, _Rows, 1>::MatrixBase() {
	this->size = 0;
}

template<typename T, size_t _Rows> 
MatrixBase<T, _Rows, 1>::MatrixBase(int _size, const T& _init) {
	this->vector.resize(_size, _init);
	this->size = _size;
}

template<typename T, size_t _Rows> 
MatrixBase<T, _Rows, 1>::MatrixBase(const T& _init) {
	this->vector.resize(_Rows, _init);
	this->size = _Rows;
}

template<typename T, size_t _Rows> 
MatrixBase<T, _Rows, 1>::MatrixBase(const MatrixBase<T, _Rows, 1>& rhs) {
	this->vector = rhs.vector;
	this->size = rhs.get_size();
}

template<typename T, size_t _Rows> 
template<size_t Other_Rows>
MatrixBase<T, _Rows, 1>::MatrixBase(const MatrixBase<T, Other_Rows, 1>& rhs) {
	this->vector = rhs.vector;
	this->size = rhs.get_size();
}

template<typename T, size_t _Rows> 
MatrixBase<T, _Rows, 1>::~MatrixBase() {}

template<typename T, size_t _Rows> 
void MatrixBase<T, _Rows, 1>::resize(const int& _size, const T& _init) {
	this->vector.resize(_size, _init);
	this->size = _size;
}

template<typename T, size_t _Rows> 
void MatrixBase<T, _Rows, 1>::resize(const int& _size) {
	this->vector.resize(_size, 0);
	this->size = _size;
}

template<typename T, size_t _Rows>
template<size_t Other_Rows>
MatrixBase<T, _Rows, 1>& MatrixBase<T, _Rows, 1>::operator=(const MatrixBase<T, Other_Rows, 1>& rhs) {
	if (&rhs == this) return *this;

	int new_size = rhs.get_size();

	this->vector.resize(new_size);

	for (int i = 0; i < new_size; i++) {
		this->vector[i] = rhs(i);
	}

	this->size = new_size;

	return *this;
}

template<typename T, size_t _Rows>
template<size_t Other_Rows>
MatrixBase<T, _Rows, 1> MatrixBase<T, _Rows, 1>::operator+(const MatrixBase<T, Other_Rows, 1>& rhs) {
	int size = rhs.get_size();
	MatrixBase<T, _Rows, 1> result(size, 0);
	if (this->size != size)
		throw std::out_of_range("Index out of bounds");

	for (int i = 0; i < size; i++) {
		result(i) = this->vector[i] + rhs(i);
	}
	return result;
}

template<typename T, size_t _Rows> 
template<size_t Other_Rows>
MatrixBase<T, _Rows, 1> MatrixBase<T, _Rows, 1>::operator-(const MatrixBase<T, Other_Rows, 1>& rhs) {
	int size = rhs.get_size();
	MatrixBase<T, _Rows, 1> result(size, 0);
	if (this->size != size)
		throw std::out_of_range("Index out of bounds");

	for (int i = 0; i < size; i++) {
		result(i) = this->vector[i] - rhs(i);
	}
	return result;
}

template<typename T, size_t _Rows>
template<size_t Other_Rows>
MatrixBase<T, _Rows, 1> MatrixBase<T, _Rows, 1>::operator*(const MatrixBase<T, Other_Rows, 1>& rhs) {
	int size = rhs.get_size();
	MatrixBase<T, _Rows, 1> result(size, 0);
	if (this->size != size)
		throw std::out_of_range("Index out of bounds");

	for (int i = 0; i < size; i++) {
		result(i) = this->vector[i] * rhs(i);
	}
	return result;
}


template<typename T, size_t _Rows>
template<size_t Other_Rows>
MatrixBase<T, _Rows, 1>& MatrixBase<T, _Rows, 1>::operator+=(const MatrixBase<T, Other_Rows, 1>& rhs) {
	int size = rhs.get_size();
	if (this->size != size)
		throw std::out_of_range("Index out of bounds");

	for (int i = 0; i < size; i++) {
		this->vector[i] += rhs(i);
	}
	return *this;
}

template<typename T, size_t _Rows>
template<size_t Other_Rows>
MatrixBase<T, _Rows, 1>& MatrixBase<T, _Rows, 1>::operator-=(const MatrixBase<T, Other_Rows, 1>& rhs) {
	int size = rhs.get_size();
	if (this->size != size)
		throw std::out_of_range("Index out of bounds");

	for (int i = 0; i < size; i++) {
		this->vector[i] -= rhs(i);
	}
	return *this;
}

template<typename T, size_t _Rows>
template<size_t Other_Rows>
MatrixBase<T, _Rows, 1>& MatrixBase<T, _Rows, 1>::operator*=(const MatrixBase<T, Other_Rows, 1>& rhs) {
	int size = rhs.get_size();
	if (this->size != size)
		throw std::out_of_range("Index out of bounds");

	for (int i = 0; i < size; i++) {
		this->vector[i] *= rhs(i);
	}
	return *this;
}

//scalar
template<typename T, size_t _Rows> 
MatrixBase<T, _Rows, 1> MatrixBase<T, _Rows, 1>::operator+(const T& rhs) {
	int size = this->size;
	MatrixBase<T, _Rows, 1> result(size, 0);

	for (int i = 0; i < size; i++) {
		result(i) = this->vector[i] + rhs;
	}

	return result;
}

template<typename T, size_t _Rows> 
MatrixBase<T, _Rows, 1> MatrixBase<T, _Rows, 1>::operator-(const T& rhs) {
	int size = this->size;
	MatrixBase<T, _Rows, 1> result(size, 0);

	for (int i = 0; i < size; i++) {
		result(i) = this->vector[i] - rhs;
	}

	return result;
}

template<typename T, size_t _Rows> 
MatrixBase<T, _Rows, 1> MatrixBase<T, _Rows, 1>::operator*(const T& rhs) {
	int size = this->size;
	MatrixBase<T, _Rows, 1> result(size, 0);

	for (int i = 0; i < size; i++) {
		result(i) = this->vector[i] * rhs;
	}

	return result;
}

template<typename T, size_t _Rows> 
MatrixBase<T, _Rows, 1> MatrixBase<T, _Rows, 1>::operator/(const T& rhs) {
	int size = this->size;
	MatrixBase<T, _Rows, 1> result(size, 0);

	for (int i = 0; i < size; i++) {
		result(i) = this->vector[i] / rhs;
	}

	return result;
}


//ref
template<typename T, size_t _Rows> 
MatrixBase<T, _Rows, 1>& MatrixBase<T, _Rows, 1>::operator+=(const T& rhs) {
	int size = this->size;

	for (int i = 0; i < size; i++) {
		 this->vector[i] += rhs;
	}

	return *this;
}

template<typename T, size_t _Rows> 
MatrixBase<T, _Rows, 1>& MatrixBase<T, _Rows, 1>::operator-=(const T& rhs) {
	int size = this->size;

	for (int i = 0; i < size; i++) {
		 this->vector[i] -= rhs;
	}

	return *this;
}

template<typename T, size_t _Rows> 
MatrixBase<T, _Rows, 1>& MatrixBase<T, _Rows, 1>::operator*=(const T& rhs) {
	int size = this->size;

	for (int i = 0; i < size; i++) {
		 this->vector[i] *= rhs;
	}

	return *this;
}

template<typename T, size_t _Rows> 
MatrixBase<T, _Rows, 1>& MatrixBase<T, _Rows, 1>::operator/=(const T& rhs) {
	int size = this->size;

	for (int i = 0; i < size; i++) {
		 this->vector[i] /= rhs;
	}

	return *this;
}

template<typename T, size_t _Rows> 
bool MatrixBase<T, _Rows, 1>::operator==(const MatrixBase<T, _Rows, 1>& rhs){
	return (*this - rhs).norm() < 0.001f;
}

template<typename T, size_t _Rows> 
MatrixBase<T, 1, _Rows> MatrixBase<T, _Rows, 1>::transpose() {
	int rows = 1;
	int cols = this->size;
	MatrixBase<T, 1, _Rows> result(0);

	for (int i = 0; i < cols; i++) {
		result(0, i) = this->vector(i);
	}
	return result;
}

template<typename T, size_t _Rows> 
T MatrixBase<T, _Rows, 1>::dot(const MatrixBase<T, _Rows, 1>& rhs) {
	T result = 0;
	if (this->size != rhs.get_size())
		throw std::out_of_range("Index out of bounds");
	for (int i = 0; i < this->size; i++) {
		result += (this->vector(i) * rhs(i));
	}
}

template<typename T, size_t _Rows> 
T& MatrixBase<T, _Rows, 1>::operator()(const int& idx) {
	return this->vector[idx];
}

template<typename T, size_t _Rows> 
T& MatrixBase<T, _Rows, 1>::operator[](const int& idx) {
	return this->vector[idx];
}


template<typename T, size_t _Rows> 
const T& MatrixBase<T, _Rows, 1>::operator()(const int& idx) const {
	return this->vector[idx];
}

template<typename T, size_t _Rows> 
const T& MatrixBase<T, _Rows, 1>::operator[](const int& idx) const {
	return this->vector[idx];
}


template<typename T, size_t _Rows> 
int MatrixBase<T, _Rows, 1>::get_size(){
	return this->size;
}


template<typename T, size_t _Rows> 
void MatrixBase<T, _Rows, 1>::push_back(T value) {
	this->vector.push_back(value);
	this->size = vector.size();
}

template<typename T, size_t _Rows> 
void MatrixBase<T, _Rows, 1>::setRandom()
{
	std::srand(std::time(0));
	for (int i = 0; i < this->size; i++) {
		this->vector[i] = ((float)std::rand() / ((float)RAND_MAX)) * 2 - 1;
	}
}

template<typename T, size_t _Rows> 
float MatrixBase<T, _Rows, 1>::norm(){
	float sum = 0;
	for (int i = 0; i < this->size; i++) {
		sum += static_cast<float>(this->vector[i] * this->vector[i]);
	}
	return std::sqrt(sum);
}

template<typename T, size_t _Rows> 
MatrixBase<T, _Rows, 1> MatrixBase<T, _Rows, 1>::square() {
	int size = this->size;
	MatrixBase<T, _Rows, 1> result(size, 0);

	for (int i = 0; i < size; i++) {
		result(i) = this->vector[i] * this->vector[i];
	}
	return result;
}

template<typename T, size_t _Rows> 
MatrixBase<T, _Rows, 1> MatrixBase<T, _Rows, 1>::sqrt() {
	int size = this->size;
	MatrixBase<T, _Rows, 1> result(size, 0);

	for (int i = 0; i < size; i++) {
		result(i) = std::sqrt(this->vector[i]);
	}
	return result;
}


template<typename T, size_t _Rows> 
MatrixBase<T, _Rows, 1> MatrixBase<T, _Rows, 1>::inv() {
	int size = this->size;
	MatrixBase<T, _Rows, 1> result(size, 0);

	for (int i = 0; i < size; i++) {
		result(i) = 1.0f / this->vector[i];
	}
	return result;
}


template<typename T, size_t _Rows> 
MatrixBase<T, _Rows, 1> MatrixBase<T, _Rows, 1>::onehot(){
	int size = this->size;
	MatrixBase<T, _Rows, 1> result(size, 0);
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
MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::MatrixBase() {
	rows = 0;
	cols = 0;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::MatrixBase(int _rows, int _cols) {
	matrix.resize(_rows);
	for (int i = 0; i < matrix.size(); i++) {
		matrix[i].resize(_cols, 0);
	}
	rows = _rows;
	cols = _cols;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::MatrixBase(int _rows, int _cols, const T& _init) {
	matrix.resize(_rows);
	for (int i = 0; i < matrix.size(); i++) {
		matrix[i].resize(_cols, _init);
	}
	rows = _rows;
	cols = _cols;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::MatrixBase(const MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& rhs) {
	matrix = rhs.matrix;
	rows = rhs.get_rows_size();
	cols = rhs.get_cols_size();
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::MatrixBase(const std::vector<std::vector<T>>& rhs) {
	this->rows = rhs.size();
	this->cols = rhs[0].size();
	this->matrix.resize(this->rows);
	for (int i = 0; i < this->rows; i++) {
		matrix[i].resize(this->cols, 0);
		for (int j = 0; j < this->cols; j++) {
			this->matrix[i][j] = rhs[i][j];
		}
	}
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::~MatrixBase() {}

template<typename T>
void MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::resize(int _rows, int _cols, T _init) {
	matrix.resize(_rows);
	for (int i = 0; i < matrix.size(); i++) {
		matrix[i].resize(_cols, _init);
	}
	rows = _rows;
	cols = _cols;
}

template<typename T>
void MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::setRandom()
{
	std::srand(std::time(0));
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			this->matrix[i][j] = ((float)std::rand() / ((float)RAND_MAX)) * 2 - 1;
		}
	}
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::operator=(const MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& rhs) {
	if (&rhs == this) return *this;

	int new_rows = rhs.get_rows_size();
	int new_cols = rhs.get_cols_size();

	this->matrix.resize(new_rows);
	for (int i = 0; i < matrix.size(); i++) {
		matrix[i].resize(new_cols);
	}

	for (int i = 0; i < new_rows; i++) {
		for (int j = 0; j < new_cols; j++) {
			matrix[i][j] = rhs(i, j);
		}
	}

	rows = new_rows;
	cols = new_cols;

	return *this;
}



template<typename T>
MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::operator+(const MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& rhs) {
	int rows = rhs.get_rows_size();
	int cols = rhs.get_cols_size();
	MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> result(rows, cols, 0);
	if (this->cols != cols || this->rows != rows)
		throw std::out_of_range("Index out of bounds");

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result(i, j) = this->matrix[i][j] + rhs(i, j);
		}
	}
	return result;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::operator-(const MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& rhs) {
	int rows = rhs.get_rows_size();
	int cols = rhs.get_cols_size();

	MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> result(rows, cols, 0);
	if (this->cols != cols || this->rows != rows)
		throw std::out_of_range("Index out of bounds");

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result(i, j) = this->matrix[i][j] - rhs(i, j);
		}
	}
	return result;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::operator*(const MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& rhs) {
	int rows = rhs.get_rows_size();
	int cols = rhs.get_cols_size();
	MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> result(rows, cols, 0);

	if (this->cols != rows)
		throw std::out_of_range("Index out of bounds");

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			for (int k = 0; k < rows; k++) {
				result(i, j) += this->matrix[i][k] * rhs(k, j);
			}
		}
	}

	return result;
}


template<typename T>
MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::operator+=(const MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& rhs) {
	int rows = rhs.get_rows_size();
	int cols = rhs.get_cols_size();
	if (this->cols != cols || this->rows != rows)
		throw std::out_of_range("Index out of bounds");

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			this->matrix[i][j] += rhs(i, j);
		}
	}

	return *this;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::operator-=(const MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& rhs) {
	int rows = rhs.get_rows_size();
	int cols = rhs.get_cols_size();
	if (this->cols != cols || this->rows != rows)
		throw std::out_of_range("Index out of bounds");

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			this->matrix[i][j] -= rhs(i, j);
		}
	}

	return *this;
}


template<typename T>
MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::operator*=(const MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& rhs) {
	MatrixBase<T> result = (*this) * rhs;
	(*this) = result;
	return *this;
}


//scalar
template<typename T>
MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::operator+(const T& rhs) {
	int rows = this->rows;
	int cols = this->cols;
	MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> result(rows, cols, 0);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result(i, j) = this->matrix[i][j] + rhs;
		}
	}

	return result;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::operator-(const T& rhs) {
	int row = this->rows;
	int cols = this->cols;
	MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> result(rows, cols, 0);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result(i, j) = this->matrix[i][j] - rhs;
		}
	}

	return result;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::operator*(const T& rhs) {
	int row = this->rows;
	int cols = this->cols;
	MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> result(rows, cols, 0);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result(i, j) = this->matrix[i][j] * rhs;
		}
	}

	return result;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::operator/(const T& rhs) {
	int row = this->rows;
	int cols = this->cols;
	MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> result(rows, cols, 0);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result(i, j) = this->matrix[i][j] / rhs;
		}
	}

	return result;
}


//ref
template<typename T>
MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::operator+=(const T& rhs) {
	int row = this->rows;
	int cols = this->cols;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			this->matrix[i][j] += rhs;
		}
	}

	return *this;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::operator-=(const T& rhs) {
	int row = this->rows;
	int cols = this->cols;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			this->matrix[i][j] -= rhs;
		}
	}

	return *this;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::operator*=(const T& rhs) {
	int row = this->rows;
	int cols = this->cols;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			this->matrix[i][j] *= rhs;
		}
	}

	return *this;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::operator/=(const T& rhs) {
	int row = this->rows;
	int cols = this->cols;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			this->matrix[i][j] /= rhs;
		}
	}

	return *this;
}

template<typename T>
MatrixBase<T,SIZE_DYNAMIC,1> MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::operator*(const MatrixBase<T,SIZE_DYNAMIC,1>& rhs) {
	if (rhs.get_size() != this->cols) {
		throw std::out_of_range("Index out of bounds");
	}
	MatrixBase<T, SIZE_DYNAMIC,1> result(this->rows, 0);
	for (int i = 0; i < this->get_rows_size(); i++) {
		for (int j = 0; j < this->get_cols_size(); j++) {
			result(i) += this->matrix[i][j] * rhs(j);
		}
	}

	return result;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::transpose() {
	int row = this->rows;
	int cols = this->cols;
	MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> result(cols, rows, 0);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result(j, i) = this->matrix[i][j];
		}
	}
	return result;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::square() {
	int rows = this->rows;
	int cols = this->cols;
	MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> result(rows, cols, 0);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result(i, j) = this->matrix[i][j] * this->matrix[i][j];
		}
	}
	return result;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::inv(){
	int rows = this->rows;
	int cols = this->cols;
	MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> result(rows, cols, 0);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result(i, j) = 1.0f / this->matrix[i][j];
		}
	}
	return result;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::sqrt(){
	int rows = this->rows;
	int cols = this->cols;
	MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> result(rows, cols, 0);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result(i, j) = std::sqrt(this->matrix[i][j]);
		}
	}
	return result;
}

template<typename T>
float MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::norm(){
	float sum = 0;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			sum += static_cast<float>(this->matrix[i][j] * this->matrix[i][j]);
		}
	}
	return std::sqrt(sum);
}

template<typename T>
T& MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::operator()(const int& row, const int& col) {
	return this->matrix[row][col];
}

template<typename T>
const T& MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::operator()(const int& row, const int& col) const {
	return this->matrix[row][col];
}

template<typename T>
int MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::get_rows_size() const {
	return this->rows;
}

template<typename T>
int MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>::get_cols_size() const {
	return this->cols;
}


template<typename T>
MatrixBase<T, SIZE_DYNAMIC,1>::MatrixBase() {
	this->size = 0;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC,1>::MatrixBase(int _size, const T& _init) {
	this->vector.resize(_size, _init);
	this->size = _size;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC,1>::MatrixBase(const MatrixBase<T, SIZE_DYNAMIC,1>& rhs) {
	this->vector = rhs.vector;
	this->size = rhs.get_size();
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC,1>::MatrixBase(const std::vector<T>& rhs) {
	this->vector.resize(rhs.size());
	this->size = rhs.size();
	for (int i = 0; i < this->size; i++) {
		this->vector[i] = rhs[i];
	}
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC,1>::~MatrixBase() {}

template<typename T>
void MatrixBase<T, SIZE_DYNAMIC,1>::resize(const int& _size, const T& _init) {
	this->vector.resize(_size, _init);
	this->size = _size;
}

template<typename T>
void MatrixBase<T, SIZE_DYNAMIC,1>::resize(const int& _size) {
	this->vector.resize(_size, 0);
	this->size = _size;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC,1>& MatrixBase<T, SIZE_DYNAMIC,1>::operator=(const MatrixBase<T, SIZE_DYNAMIC,1>& rhs) {
	if (&rhs == this) return *this;

	int new_size = rhs.get_size();

	this->vector.resize(new_size);

	for (int i = 0; i < new_size; i++) {
		this->vector[i] = rhs(i);
	}

	this->size = new_size;

	return *this;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC,1> MatrixBase<T, SIZE_DYNAMIC,1>::operator+(const MatrixBase<T, SIZE_DYNAMIC,1>& rhs) {
	int size = rhs.get_size();
	MatrixBase<T, SIZE_DYNAMIC,1> result(size, 0);
	if (this->size != size)
		throw std::out_of_range("Index out of bounds");

	for (int i = 0; i < size; i++) {
		result(i) = this->vector[i] + rhs(i);
	}
	return result;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC,1> MatrixBase<T, SIZE_DYNAMIC,1>::operator-(const MatrixBase<T, SIZE_DYNAMIC,1>& rhs) {
	int size = rhs.get_size();
	MatrixBase<T, SIZE_DYNAMIC,1> result(size, 0);
	if (this->size != size)
		throw std::out_of_range("Index out of bounds");

	for (int i = 0; i < size; i++) {
		result(i) = this->vector[i] - rhs(i);
	}
	return result;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC,1> MatrixBase<T, SIZE_DYNAMIC,1>::operator*(const MatrixBase<T, SIZE_DYNAMIC,1>& rhs) {
	int size = rhs.get_size();
	MatrixBase<T, SIZE_DYNAMIC,1> result(size, 0);
	if (this->size != size)
		throw std::out_of_range("Index out of bounds");

	for (int i = 0; i < size; i++) {
		result(i) = this->vector[i] * rhs(i);
	}
	return result;
}


template<typename T>
MatrixBase<T, SIZE_DYNAMIC,1>& MatrixBase<T, SIZE_DYNAMIC,1>::operator+=(const MatrixBase<T, SIZE_DYNAMIC,1>& rhs) {
	int size = rhs.get_size();
	if (this->size != size)
		throw std::out_of_range("Index out of bounds");

	for (int i = 0; i < size; i++) {
		this->vector[i] += rhs(i);
	}
	return *this;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC,1>& MatrixBase<T, SIZE_DYNAMIC,1>::operator-=(const MatrixBase<T, SIZE_DYNAMIC,1>& rhs) {
	int size = rhs.get_size();
	if (this->size != size)
		throw std::out_of_range("Index out of bounds");

	for (int i = 0; i < size; i++) {
		this->vector[i] -= rhs(i);
	}
	return *this;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC,1>& MatrixBase<T, SIZE_DYNAMIC,1>::operator*=(const MatrixBase<T, SIZE_DYNAMIC,1>& rhs) {
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
MatrixBase<T, SIZE_DYNAMIC,1> MatrixBase<T, SIZE_DYNAMIC,1>::operator+(const T& rhs) {
	int size = this->size;
	MatrixBase<T, SIZE_DYNAMIC,1> result(size, 0);

	for (int i = 0; i < size; i++) {
		result(i) = this->vector[i] + rhs;
	}

	return result;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC,1> MatrixBase<T, SIZE_DYNAMIC,1>::operator-(const T& rhs) {
	int size = this->size;
	MatrixBase<T, SIZE_DYNAMIC,1> result(size, 0);

	for (int i = 0; i < size; i++) {
		result(i) = this->vector[i] - rhs;
	}

	return result;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC,1> MatrixBase<T, SIZE_DYNAMIC,1>::operator*(const T& rhs) {
	int size = this->size;
	MatrixBase<T, SIZE_DYNAMIC,1> result(size, 0);

	for (int i = 0; i < size; i++) {
		result(i) = this->vector[i] * rhs;
	}

	return result;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC,1> MatrixBase<T, SIZE_DYNAMIC,1>::operator/(const T& rhs) {
	int size = this->size;
	MatrixBase<T, SIZE_DYNAMIC,1> result(size, 0);

	for (int i = 0; i < size; i++) {
		result(i) = this->vector[i] / rhs;
	}

	return result;
}


//ref
template<typename T>
MatrixBase<T, SIZE_DYNAMIC,1>& MatrixBase<T, SIZE_DYNAMIC,1>::operator+=(const T& rhs) {
	int size = this->size;

	for (int i = 0; i < size; i++) {
		 this->vector[i] += rhs;
	}

	return *this;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC,1>& MatrixBase<T, SIZE_DYNAMIC,1>::operator-=(const T& rhs) {
	int size = this->size;

	for (int i = 0; i < size; i++) {
		 this->vector[i] -= rhs;
	}

	return *this;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC,1>& MatrixBase<T, SIZE_DYNAMIC,1>::operator*=(const T& rhs) {
	int size = this->size;

	for (int i = 0; i < size; i++) {
		 this->vector[i] *= rhs;
	}

	return *this;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC,1>& MatrixBase<T, SIZE_DYNAMIC,1>::operator/=(const T& rhs) {
	int size = this->size;

	for (int i = 0; i < size; i++) {
		 this->vector[i] /= rhs;
	}

	return *this;
}

template<typename T>
bool MatrixBase<T, SIZE_DYNAMIC,1>::operator==(const MatrixBase<T, SIZE_DYNAMIC,1>& rhs){
	return (*this - rhs).norm() < 0.001f;
}

template<typename T>
MatrixBase<T> MatrixBase<T, SIZE_DYNAMIC,1>::transpose() {
	int rows = 1;
	int cols = this->size;
	MatrixBase<T, 1, SIZE_DYNAMIC> result(rows, rows, 0);

	for (int i = 0; i < cols; i++) {
		result(0, i) = this->vector(i);
	}
	return result;
}

template<typename T>
T MatrixBase<T, SIZE_DYNAMIC,1>::dot(const MatrixBase<T, SIZE_DYNAMIC,1>& rhs) {
	T result = 0;
	if (this->size != rhs.get_size())
		throw std::out_of_range("Index out of bounds");
	for (int i = 0; i < this->size; i++) {
		result += (this->vector(i) * rhs(i));
	}
}

template<typename T>
T& MatrixBase<T, SIZE_DYNAMIC,1>::operator()(const int& idx) {
	return this->vector[idx];
}

template<typename T>
T& MatrixBase<T, SIZE_DYNAMIC,1>::operator[](const int& idx) {
	return this->vector[idx];
}


template<typename T>
const T& MatrixBase<T, SIZE_DYNAMIC,1>::operator()(const int& idx) const {
	return this->vector[idx];
}

template<typename T>
T& MatrixBase<T, SIZE_DYNAMIC,1>::operator()(const int& begin, const int& end)
{
	MatrixBase<T, SIZE_DYNAMIC,1> result = *this;
	result.resize(end - begin);
	result.vector = result.vector(begin, end);
	return result;
}

template<typename T>
const T& MatrixBase<T, SIZE_DYNAMIC,1>::operator[](const int& idx) const {
	return this->vector[idx];
}


template<typename T>
int MatrixBase<T, SIZE_DYNAMIC,1>::get_size() const {
	return this->size;
}


template<typename T>
void MatrixBase<T, SIZE_DYNAMIC,1>::push_back(T value) {
	this->vector.push_back(value);
	this->size = vector.size();
}

template<typename T>
void MatrixBase<T, SIZE_DYNAMIC,1>::setRandom()
{
	std::srand(std::time(0));
	for (int i = 0; i < this->size; i++) {
		this->vector[i] = ((float)std::rand() / ((float)RAND_MAX)) * 2 - 1;
	}
}

template<typename T>
float MatrixBase<T, SIZE_DYNAMIC,1>::norm(){
	float sum = 0;
	for (int i = 0; i < this->size; i++) {
		sum += static_cast<float>(this->vector[i] * this->vector[i]);
	}
	return std::sqrt(sum);
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC,1> MatrixBase<T, SIZE_DYNAMIC,1>::square() {
	int size = this->size;
	MatrixBase<T, SIZE_DYNAMIC,1> result(size, 0);

	for (int i = 0; i < size; i++) {
		result(i) = this->vector[i] * this->vector[i];
	}
	return result;
}

template<typename T>
MatrixBase<T, SIZE_DYNAMIC,1> MatrixBase<T, SIZE_DYNAMIC,1>::sqrt() {
	int size = this->size;
	MatrixBase<T, SIZE_DYNAMIC,1> result(size, 0);

	for (int i = 0; i < size; i++) {
		result(i) = std::sqrt(this->vector[i]);
	}
	return result;
}


template<typename T>
MatrixBase<T, SIZE_DYNAMIC,1> MatrixBase<T, SIZE_DYNAMIC,1>::inv() {
	int size = this->size;
	MatrixBase<T, SIZE_DYNAMIC,1> result(size, 0);

	for (int i = 0; i < size; i++) {
		result(i) = 1.0f / this->vector[i];
	}
	return result;
}


template<typename T>
MatrixBase<T, SIZE_DYNAMIC,1> MatrixBase<T, SIZE_DYNAMIC,1>::onehot(){
	int size = this->size;
	MatrixBase<T, SIZE_DYNAMIC,1> result(size, 0);
	float _max = *std::max_element(this->vector.begin(), this->vector.end());
	for(int i = 0; i < size; i++){
		if (_max == this->vector[i]){
			result[i] = 1;
			break;
		}
	}
	return result;
}


#endif