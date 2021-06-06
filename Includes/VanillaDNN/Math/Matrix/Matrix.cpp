#ifndef VANILLA_DNN_MATRIX_CPP
#define VANILLA_DNN_MATRIX_CPP

#include <VanillaDNN/Math/Matrix/Matrix.hpp>

template<typename T>
Matrix<T>::Matrix() {
	rows = 0;
	cols = 0;
}

template<typename T>
Matrix<T>::Matrix(int _rows, int _cols) {
	matrix.resize(_rows);
	for (int i = 0; i < matrix.size(); i++) {
		matrix[i].resize(_cols, 0);
	}
	rows = _rows;
	cols = _cols;
}

template<typename T>
Matrix<T>::Matrix(int _rows, int _cols, const T& _init) {
	matrix.resize(_rows);
	for (int i = 0; i < matrix.size(); i++) {
		matrix[i].resize(_cols, _init);
	}
	rows = _rows;
	cols = _cols;
}

template<typename T>
Matrix<T>::Matrix(const Matrix<T>& rhs) {
	matrix = rhs.matrix;
	rows = rhs.get_rows_size();
	cols = rhs.get_cols_size();
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
Matrix<T>::~Matrix() {}

template<typename T>
void Matrix<T>::resize(int _rows, int _cols, T _init) {
	matrix.resize(_rows);
	for (int i = 0; i < matrix.size(); i++) {
		matrix[i].resize(_cols, _init);
	}
	rows = _rows;
	cols = _cols;
}

template<typename T>
void Matrix<T>::setRandom()
{
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			this->matrix[i][j] = ((float)rand() / (RAND_MAX)) * 2 - 1;
		}
	}
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& rhs) {
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
	Matrix result = (*this) * rhs;
	(*this) = result;
	return *this;
}


//scalar
template<typename T>
Matrix<T> Matrix<T>::operator+(const T& rhs) {
	int row = this->rows;
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
	int row = this->rows;
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
	int row = this->rows;
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
	int row = this->rows;
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
Matrix<T>& Matrix<T>::operator-=(const T& rhs) {
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
Matrix<T>& Matrix<T>::operator*=(const T& rhs) {
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
Matrix<T>& Matrix<T>::operator/=(const T& rhs) {
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
Vector<T> Matrix<T>::operator*(const Vector<T>& rhs) {
	if (rhs.get_size() != this->cols) {
		throw std::out_of_range("Index out of bounds");
	}
	Vector<T> result(this->rows, 0);
	for (int i = 0; i < this->get_rows_size(); i++) {
		for (int j = 0; j < this->get_cols_size(); j++) {
			result(i) += this->matrix[i][j] * rhs(j);
		}
	}

	return result;
}

template<typename T>
Matrix<T> Matrix<T>::transpose() {
	int row = this->rows;
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
	int row = this->rows;
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
float Matrix<T>::norm(){
	float sum = 0;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			sum += static_cast<float>(this->matrix[i][j] * this->matrix[i][j]);
		}
	}
	return sqrt(sum);
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