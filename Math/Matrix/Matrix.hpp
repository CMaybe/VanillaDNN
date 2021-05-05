#ifndef VANILLA_DNN_MATRIX_HPP
#define VANILLA_DNN_MATRIX_HPP

#include <vector>
#include <iostream>
#include <stdexcept>

template<typename T> class Vector;

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
		Matrix<T> operator+(const Matrix<T>& rhs);
		Matrix<T> operator-(const Matrix<T>& rhs);
		Matrix<T> operator*(const Matrix<T>& rhs);
		Matrix<T>& operator+=(const Matrix<T>& rhs);
		Matrix<T>& operator-=(const Matrix<T>& rhs);
		Matrix<T>& operator*=(const Matrix<T>& rhs);
	
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
	
		//cout overloading 
		template<typename U>
		friend std::ostream & operator << (std::ostream &out, const Matrix<U> &m);

		int get_rows_size() const;
		int get_cols_size() const;
	
	
};

template<typename U>
std::ostream & operator << (std::ostream &out,  const Matrix<U> &m)
{
	int rows = m.get_rows_size();
	int cols = m.get_cols_size();
    out << "rows : " << m.get_rows_size() <<'\n'<< "cols : " << m.get_cols_size()<<'\n';
    for(int i = 0;i<rows;i++){
		for(int j = 0;j < cols; j++){
			out<<m(i,j)<<'\t';
		}
		out<<'\n';
	}
    return out;
}


#include"Matrix.cpp"

#endif