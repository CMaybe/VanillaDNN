#ifndef VANILLA_DNN_MATRIX_CPP
#define VANILLA_DNN_MATRIX_CPP

#include <VanillaDNN/Math/Matrix/Matrix.hpp>

template<typename T>
Matrix<T>::Matrix() {
	this->rows = 0;
	this->cols = 0;
}

template<typename T>
Matrix<T>::Matrix(const int& _rows, const int& _cols) {
	this->rows = _rows;
	this->cols = _cols;
	this->resize(_rows, _cols);
}

template<typename T>
Matrix<T>::Matrix(const int& _rows, const int&_cols, const T& _init) {
	this->rows = _rows;
	this->cols = _cols;
	this->resize(_rows, _cols, _init);
}

template<typename T>
Matrix<T>::Matrix(const Matrix<T>& rhs) {
	this->matrix = rhs.matrix;
	this->rows = rhs.get_rows_size();
	this->cols = rhs.get_cols_size();
}

template<typename T>
Matrix<T>::Matrix(const Vector<T>& rhs) {
	this->rows = rhs.get_size();
	this->cols = 1;
	this->resize();
	
	for (int i = 0; i < this->rows; i++) {
		matrix[i][0] = rhs[i];
	}
}

template<typename T>
Matrix<T>::Matrix(const std::vector<std::vector<T>>& rhs) {
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
Matrix<T>::~Matrix() {
	this->clear();
}

template<typename T>
void Matrix<T>::resize(int _rows, int _cols, T _init) {
	this->clear();
	std::vector<std::vector<T>>(_rows).swap(this->matrix);
	for (int i = 0; i < _rows; i++){
		std::vector<T>(_cols, _init).swap(this->matrix[i]);
	}
	this->rows = _rows;
	this->cols = _cols;
	
	return;
}

template<typename T>
void Matrix<T>::resize() {
	this->clear();
	std::vector<std::vector<T>>(this->rows).swap(this->matrix);
	for (int i = 0; i < this->rows; i++){
		this->matrix[i] = std::vector<T>(this->cols, 0);
	}
	return;
}

template<typename T>
void Matrix<T>::setRandom()
{
	std::srand(std::time(0));
	for (int i = 0; i < this->rows; i++) {
		for (int j = 0; j < this->cols; j++) {
			this->matrix[i][j] = ((float)std::rand() / ((float)RAND_MAX)) * 2 - 1;
		}
	}
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& rhs) {
	if (&rhs == this) return *this;
	this->clear();
	this->rows = rhs.get_rows_size();
	this->cols = rhs.get_cols_size();
	this->resize();
	
	for (int i = 0; i < this->rows; i++) {
		for (int j = 0; j < this->cols; j++) {
			matrix[i][j] = rhs(i, j);
		}
	}


	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(const Vector<T>& rhs) {
	this->clear();
	this->rows = rhs.get_size();
	this->cols = 1;
	this->resize();
	
	for (int i = 0; i < this->rows; i++) {
		matrix[i][0] = rhs(i);
	}

	return *this;
}

template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& rhs) {
	int rows = rhs.get_rows_size();
	int cols = rhs.get_cols_size();
	Matrix<T> result(rows, cols, 0);
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
Matrix<T> Matrix<T>::operator-(const Matrix<T>& rhs) {
	int rows = rhs.get_rows_size();
	int cols = rhs.get_cols_size();

	Matrix<T> result(rows, cols, 0);
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
Matrix<T> Matrix<T>::operator*(const Matrix<T>& rhs) {
	int rows = rhs.get_rows_size();
	int cols = rhs.get_cols_size();

	Matrix<T> result(rows, cols, 0);
	if (this->cols != cols || this->rows != rows)
		throw std::out_of_range("Index out of bounds");

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result(i, j) = this->matrix[i][j] * rhs(i, j);
		}
	}
	return result;
}



template<typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& rhs) {
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
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& rhs) {
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
Matrix<T>& Matrix<T>::operator*=(const Matrix<T>& rhs) {
	int rows = rhs.get_rows_size();
	int cols = rhs.get_cols_size();
	if (this->cols != cols || this->rows != rows)
		throw std::out_of_range("Index out of bounds");

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			this->matrix[i][j] *= rhs(i, j);
		}
	}

	return *this;
}


//scalar
template<typename T>
Matrix<T> Matrix<T>::operator+(const T& rhs) {
	int rows = this->rows;
	int cols = this->cols;
	Matrix<T> result(rows, cols, 0);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result(i, j) = this->matrix[i][j] + rhs;
		}
	}

	return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(const T& rhs) {
	int rows = this->rows;
	int cols = this->cols;
	Matrix<T> result(rows, cols, 0);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result(i, j) = this->matrix[i][j] - rhs;
		}
	}

	return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const T& rhs) {
	int rows = this->rows;
	int cols = this->cols;
	Matrix<T> result(rows, cols, 0);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result(i, j) = this->matrix[i][j] * rhs;
		}
	}

	return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator/(const T& rhs) {
	int rows = this->rows;
	int cols = this->cols;
	Matrix<T> result(rows, cols, 0);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result(i, j) = this->matrix[i][j] / rhs;
		}
	}

	return result;
}


//ref
template<typename T>
Matrix<T>& Matrix<T>::operator+=(const T& rhs) {
	int rows = this->rows;
	int cols = this->cols;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			this->matrix[i][j] += rhs;
		}
	}

	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator-=(const T& rhs) {
	int rows = this->rows;
	int cols = this->cols;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			this->matrix[i][j] -= rhs;
		}
	}

	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator*=(const T& rhs) {
	int rows = this->rows;
	int cols = this->cols;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			this->matrix[i][j] *= rhs;
		}
	}

	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator/=(const T& rhs) {
	int rows = this->rows;
	int cols = this->cols;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			this->matrix[i][j] /= rhs;
		}
	}

	return *this;
}


template<typename T>
Matrix<T> Matrix<T>::dot(const Matrix<T>& rhs) {
	int rows = this->rows;
	int cols = rhs.get_cols_size();
	Matrix<T> result(rows, cols, 0);

	if (this->cols != rhs.get_rows_size())
		throw std::out_of_range("Index out of bounds");

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			for (int k = 0; k < this->cols; k++) {
				result(i, j) += this->matrix[i][k] * rhs(k, j);
			}
		}
	}

	return result;
}

template<typename T>
Vector<T> Matrix<T>::dot(const Vector<T>& rhs) {
	if (rhs.get_size() != this->cols) {
		throw std::out_of_range("Index out of bounds");
	}
	
	Vector<T> result(this->rows, 0);
	for (int i = 0; i < this->rows; i++) {
		for (int j = 0; j < this->cols; j++) {
			result(i) += this->matrix[i][j] * rhs(j);
		}
	}

	return result;
}

template<typename T>
Matrix<T> Matrix<T>::transpose(){
	int rows = this->rows;
	int cols = this->cols;
	Matrix<T> result(cols, rows, 0);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result(j, i) = this->matrix[i][j];
		}
	}
	return result;
}

template<typename T>
Matrix<T> Matrix<T>::square() {
	int rows = this->rows;
	int cols = this->cols;
	Matrix<T> result(rows, cols, 0);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result(i, j) = this->matrix[i][j] * this->matrix[i][j];
		}
	}
	return result;
}

template<typename T>
Matrix<T> Matrix<T>::inv(){
	int rows = this->rows;
	int cols = this->cols;
	Matrix<T> result(rows, cols, 0);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result(i, j) = 1.0f / this->matrix[i][j];
		}
	}
	return result;
}

template<typename T>
Matrix<T> Matrix<T>::sqrt(){
	int rows = this->rows;
	int cols = this->cols;
	Matrix<T> result(rows, cols, 0);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result(i, j) = std::sqrt(this->matrix[i][j]);
		}
	}
	return result;
}

template<typename T>
float Matrix<T>::norm(){
	float sum = 0;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			sum += static_cast<float>(this->matrix[i][j] * this->matrix[i][j]);
		}
	}
	return std::sqrt(sum);
}

template<typename T>
Matrix<T> Matrix<T>::clip(const T& _min, const T& _max){
	int rows = this->rows;
	int cols = this->cols;
	Matrix<T> result(rows, cols, 0);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (_max < this->matrix[i][j]) result(i, j) = _max;
			else if (_min > this->matrix[i][j]) result(i, j) = _min;
			else result(i, j) = this->matrix[i][j];
		}
	}
	return result;
}

template<typename T>
T Matrix<T>::sum(){
	T result = 0;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result += this->matrix[i][j];
		}
	}
	return result;
}
template<typename T>
void Matrix<T>::clear(){
	for (int i = 0; i < this->matrix.size(); i++){
		std::vector<T>().swap(this->matrix[i]);
	}
	std::vector<std::vector<T>>().swap(this->matrix);
	return;
}


template<typename T>
T& Matrix<T>::operator()(const int& row, const int& col) {
	return this->matrix[row][col];
}

template<typename T>
const T& Matrix<T>::operator()(const int& row, const int& col) const {
	return this->matrix[row][col];
}

template<typename T>
int Matrix<T>::get_rows_size() const {
	return this->rows;
}

template<typename T>
int Matrix<T>::get_cols_size() const {
	return this->cols;
}

#endif