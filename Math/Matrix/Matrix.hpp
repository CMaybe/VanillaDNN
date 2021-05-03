#pragma once

#include<vector>
#include"Vector/Vector.hpp"

template<typename T> class Matrix{
	private:
		std::vector<std::vector<T>> matrix;
		int rows;
		int cols;
	
	public:
		Matrix(int _rows,int _cols, const T& _init);
		Matrix(const Matrix<T>& rhs);
		Matrix<T>& operator=(const Matrix<T>& rhs);
	
		virtual ~Matrix();
	
	
		//operator
		Matrix<T>& operator+=(const Matrix<T>& rhs);
		Matrix<T> operator=(const Matrix<T>& rhs);
	
		Matrix<T>& operator-=(const Matrix<T>& rhs);
		Matrix<T> operator-(const Matrix<T>& rhs);
	
		Matrix<T>& operator*=(const Matrix<T>& rhs);
		Matrix<T> operator*(const Matrix<T>& rhs);
	
		Matrix<T> transpose();
	
	
		//scalar	
		Matrix<T> operator+(const T& rhs);
		Matrix<T> operator-(const T& rhs);
		Matrix<T> operator*(const T& rhs);
		Matrix<T> operator/(const T& rhs);
		Matrix<T>& operator+=(const T& rhs);
		Matrix<T>& operator-=(const T& rhs);
		Matrix<T>& operator*=(const T& rhs);
		Matrix<T>& operator/=(const T& rhs);
	
		Vector<T> operator*(const Vector<T>& rhs);
	
		T& operator()(const int& row, const int& col);
		const T& operator()(const int& row, const int& col) const;
	
		int get_rows_size() const;
		int get_cols_size() const;
	
	
};

#include"Matrix.cpp"