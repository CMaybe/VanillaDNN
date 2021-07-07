#ifndef VANILLA_DNN_MATRIX_HPP
#define VANILLA_DNN_MATRIX_HPP

#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>

template<typename T = float,size_t _Rows=0,size_t _Cols=0> class Matrix {
private:
	std::vector<std::vector<T>> matrix;
	int rows;
	int cols;

public:
	Matrix();
	Matrix(int _rows, int _cols, const T& _init=0);
	Matrix(const T& _init);
	Matrix(const Matrix<T, _Rows, _Cols>& rhs);
	Matrix(const std::vector<std::vector<T>>& rhs);
	template<size_t Other_Rows,size_t Other_Cols>
	Matrix(const Matrix<T, Other_Rows, Other_Cols>& rhs);

	
	virtual ~Matrix();


	//operator
	template<size_t Other_Rows,size_t Other_Cols>
	Matrix<T, _Rows, _Cols>& operator=(const Matrix<T, Other_Rows, Other_Cols>& rhs);
	template<size_t Other_Rows,size_t Other_Cols>
	Matrix<T, _Rows, _Cols> operator+(const Matrix<T, Other_Rows, Other_Cols>& rhs);
	template<size_t Other_Rows,size_t Other_Cols>
	Matrix<T, _Rows, _Cols> operator-(const Matrix<T, Other_Rows, Other_Cols>& rhs);
	template<size_t Other_Rows,size_t Other_Cols>
	Matrix<T, _Rows, Other_Cols> operator*(const Matrix<T, Other_Rows, Other_Cols>& rhs);
	template<size_t Other_Rows,size_t Other_Cols>
	Matrix<T, _Rows, _Cols>& operator+=(const Matrix<T, Other_Rows, Other_Cols>& rhs);
	template<size_t Other_Rows,size_t Other_Cols>
	Matrix<T, _Rows, _Cols>& operator-=(const Matrix<T, Other_Rows, Other_Cols>& rhs);
	template<size_t Other_Rows,size_t Other_Cols>
	Matrix<T, _Rows, Other_Cols>& operator*=(const Matrix<T, Other_Rows, Other_Cols>& rhs);

	Matrix<T, _Cols, _Rows> transpose();
	Matrix<T, _Rows, _Cols> square();
	Matrix<T, _Rows, _Cols> sqrt();
	Matrix<T, _Rows, _Cols> inv();

	//scalar	
	Matrix<T, _Rows, _Cols> operator+(const T& rhs);
	Matrix<T, _Rows, _Cols> operator-(const T& rhs);
	Matrix<T, _Rows, _Cols> operator*(const T& rhs);
	Matrix<T, _Rows, _Cols> operator/(const T& rhs);
	Matrix<T, _Rows, _Cols>& operator+=(const T& rhs);
	Matrix<T, _Rows, _Cols>& operator-=(const T& rhs);
	Matrix<T, _Rows, _Cols>& operator*=(const T& rhs);
	Matrix<T, _Rows, _Cols>& operator/=(const T& rhs);
	


	T& operator()(const int& row, const int& col);
	const T& operator()(const int& row, const int& col) const;
	
	

	//cout overloading 
	template<typename U>
	friend std::ostream& operator << (std::ostream& out, const Matrix<U>& m);

	int get_rows_size() const;
	int get_cols_size() const;
	void resize(int _rows, int _cols, T _init = 0);
	void setRandom();
	float norm();

};


template<typename T, size_t _Rows> class Matrix<T, _Rows,1> {
private:
	std::vector<T> vector;
	int size;

public:
	Matrix();
	Matrix(const T& _init);
	Matrix(int _size, const T& _init =0);
	template<size_t Other_Rows>
	Matrix(const Matrix<T,Other_Rows,1>& rhs);
	Matrix(const Matrix<T,_Rows,1>& rhs);
	Matrix(const std::vector<T>& rhs);
	Matrix<T,_Rows,1>& operator=(const Matrix<T,_Rows,1>& rhs);
	bool operator==(const Matrix<T,_Rows,1>& rhs);
	
	virtual ~Matrix();


	//operator
	template<size_t Other_Rows>
	Matrix<T, _Rows, 1>& operator=(const Matrix<T, Other_Rows, 1>& rhs);
	template<size_t Other_Rows>
	Matrix<T, _Rows, 1> operator+(const Matrix<T, Other_Rows, 1>& rhs);
	template<size_t Other_Rows>
	Matrix<T, _Rows, 1> operator-(const Matrix<T, Other_Rows, 1>& rhs);
	template<size_t Other_Rows>
	Matrix<T, _Rows, 1> operator*(const Matrix<T, Other_Rows, 1>& rhs);
	template<size_t Other_Rows>
	Matrix<T, _Rows, 1>& operator+=(const Matrix<T, Other_Rows, 1>& rhs);
	template<size_t Other_Rows>
	Matrix<T, _Rows, 1>& operator-=(const Matrix<T, Other_Rows, 1>& rhs);
	template<size_t Other_Rows>
	Matrix<T, _Rows, 1>& operator*=(const Matrix<T, Other_Rows, 1>& rhs);

	Matrix<T, 1, _Rows> transpose();
	Matrix<T, _Rows, 1> square();
	Matrix<T, _Rows, 1> sqrt();
	Matrix<T, _Rows, 1> inv();

	//scalar	
	Matrix<T, _Rows, 1> operator+(const T& rhs);
	Matrix<T, _Rows, 1> operator-(const T& rhs);
	Matrix<T, _Rows, 1> operator*(const T& rhs);
	Matrix<T, _Rows, 1> operator/(const T& rhs);
	Matrix<T, _Rows, 1>& operator+=(const T& rhs);
	Matrix<T, _Rows, 1>& operator-=(const T& rhs);
	Matrix<T, _Rows, 1>& operator*=(const T& rhs);
	Matrix<T, _Rows, 1>& operator/=(const T& rhs);
	
	T dot(const Matrix<T, _Rows, 1>& rhs);


	

	

	
	T& operator()(const int& idx);
	const T& operator()(const int& idx) const;

	T& operator[](const int& idx);
	const T& operator[](const int& idx) const;
	
	// Matrix<T> onehot();
	

	Matrix<T,_Rows, 1> onehot();
	int get_size();
	void push_back(T value);
	void setRandom();
	void resize(const int& _size);
	void resize(const int& _size, const T& _init);
	float norm();

};


template<typename U>
std::ostream& operator << (std::ostream& out, const Matrix<U>& m)
{
	int rows = m.get_rows_size();
	int cols = m.get_cols_size();
	out << "rows : " << m.get_rows_size() << '\n' << "cols : " << m.get_cols_size() << '\n';
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			out << m(i, j) << '\t';
		}
		out << '\n';
	}
	return out;
}

#include <VanillaDNN/Math/Matrix/Matrix.cpp>
#endif