#ifndef VANILLA_DNN_MATRIX_HPP
#define VANILLA_DNN_MATRIX_HPP

#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>

#define SIZE_DYNAMIC 0

template<typename T,size_t _Rows=SIZE_DYNAMIC,size_t _Cols=SIZE_DYNAMIC> 
class MatrixBase {
private:
	std::vector<std::vector<T>> matrix;
	int rows;
	int cols;

public:
	MatrixBase();
	MatrixBase(int _rows, int _cols, const T& _init=0);
	MatrixBase(const T& _init);
	MatrixBase(const MatrixBase<T, _Rows, _Cols>& rhs);
	MatrixBase(const std::vector<std::vector<T>>& rhs);
	template<size_t Other_Rows,size_t Other_Cols>
	MatrixBase(const MatrixBase<T, Other_Rows, Other_Cols>& rhs);

	
	virtual ~MatrixBase();


	//operator
	template<size_t Other_Rows,size_t Other_Cols>
	MatrixBase<T, _Rows, _Cols>& operator=(const MatrixBase<T, Other_Rows, Other_Cols>& rhs);
	template<size_t Other_Rows,size_t Other_Cols>
	MatrixBase<T, _Rows, _Cols> operator+(const MatrixBase<T, Other_Rows, Other_Cols>& rhs);
	template<size_t Other_Rows,size_t Other_Cols>
	MatrixBase<T, _Rows, _Cols> operator-(const MatrixBase<T, Other_Rows, Other_Cols>& rhs);
	template<size_t Other_Rows,size_t Other_Cols>
	MatrixBase<T, _Rows, Other_Cols> operator*(const MatrixBase<T, Other_Rows, Other_Cols>& rhs);
	template<size_t Other_Rows,size_t Other_Cols>
	MatrixBase<T, _Rows, _Cols>& operator+=(const MatrixBase<T, Other_Rows, Other_Cols>& rhs);
	template<size_t Other_Rows,size_t Other_Cols>
	MatrixBase<T, _Rows, _Cols>& operator-=(const MatrixBase<T, Other_Rows, Other_Cols>& rhs);
	template<size_t Other_Rows,size_t Other_Cols>
	MatrixBase<T, _Rows, Other_Cols>& operator*=(const MatrixBase<T, Other_Rows, Other_Cols>& rhs);

	MatrixBase<T, _Cols, _Rows> transpose();
	MatrixBase<T, _Rows, _Cols> square();
	MatrixBase<T, _Rows, _Cols> sqrt();
	MatrixBase<T, _Rows, _Cols> inv();

	//scalar	
	MatrixBase<T, _Rows, _Cols> operator+(const T& rhs);
	MatrixBase<T, _Rows, _Cols> operator-(const T& rhs);
	MatrixBase<T, _Rows, _Cols> operator*(const T& rhs);
	MatrixBase<T, _Rows, _Cols> operator/(const T& rhs);
	MatrixBase<T, _Rows, _Cols>& operator+=(const T& rhs);
	MatrixBase<T, _Rows, _Cols>& operator-=(const T& rhs);
	MatrixBase<T, _Rows, _Cols>& operator*=(const T& rhs);
	MatrixBase<T, _Rows, _Cols>& operator/=(const T& rhs);
	


	T& operator()(const int& row, const int& col);
	const T& operator()(const int& row, const int& col) const;
	
	

	//cout overloading 
	template<typename U>
	friend std::ostream& operator << (std::ostream& out, const MatrixBase<U,_Rows,_Cols>& m);

	int get_rows_size() const;
	int get_cols_size() const;
	void resize(int _rows, int _cols, T _init = 0);
	void setRandom();
	float norm();

};


template<typename T, size_t _Rows> 
class MatrixBase<T, _Rows,1> {
private:
	std::vector<T> vector;
	int size;

public:
	MatrixBase();
	MatrixBase(const T& _init);
	MatrixBase(int _size, const T& _init =0);
	template<size_t Other_Rows>
	MatrixBase(const MatrixBase<T,Other_Rows,1>& rhs);
	MatrixBase(const MatrixBase<T,_Rows,1>& rhs);
	MatrixBase(const std::vector<T>& rhs);
	MatrixBase<T,_Rows,1>& operator=(const MatrixBase<T,_Rows,1>& rhs);
	bool operator==(const MatrixBase<T,_Rows,1>& rhs);
	
	virtual ~MatrixBase();


	//operator
	template<size_t Other_Rows>
	MatrixBase<T, _Rows, 1>& operator=(const MatrixBase<T, Other_Rows, 1>& rhs);
	template<size_t Other_Rows>
	MatrixBase<T, _Rows, 1> operator+(const MatrixBase<T, Other_Rows, 1>& rhs);
	template<size_t Other_Rows>
	MatrixBase<T, _Rows, 1> operator-(const MatrixBase<T, Other_Rows, 1>& rhs);
	template<size_t Other_Rows>
	MatrixBase<T, _Rows, 1> operator*(const MatrixBase<T, Other_Rows, 1>& rhs);
	template<size_t Other_Rows>
	MatrixBase<T, _Rows, 1>& operator+=(const MatrixBase<T, Other_Rows, 1>& rhs);
	template<size_t Other_Rows>
	MatrixBase<T, _Rows, 1>& operator-=(const MatrixBase<T, Other_Rows, 1>& rhs);
	template<size_t Other_Rows>
	MatrixBase<T, _Rows, 1>& operator*=(const MatrixBase<T, Other_Rows, 1>& rhs);

	MatrixBase<T, 1, _Rows> transpose();
	MatrixBase<T, _Rows, 1> square();
	MatrixBase<T, _Rows, 1> sqrt();
	MatrixBase<T, _Rows, 1> inv();

	//scalar	
	MatrixBase<T, _Rows, 1> operator+(const T& rhs);
	MatrixBase<T, _Rows, 1> operator-(const T& rhs);
	MatrixBase<T, _Rows, 1> operator*(const T& rhs);
	MatrixBase<T, _Rows, 1> operator/(const T& rhs);
	MatrixBase<T, _Rows, 1>& operator+=(const T& rhs);
	MatrixBase<T, _Rows, 1>& operator-=(const T& rhs);
	MatrixBase<T, _Rows, 1>& operator*=(const T& rhs);
	MatrixBase<T, _Rows, 1>& operator/=(const T& rhs);
	
	T dot(const MatrixBase<T, _Rows, 1>& rhs);

	
	T& operator()(const int& idx);
	const T& operator()(const int& idx) const;

	T& operator[](const int& idx);
	const T& operator[](const int& idx) const;
	
	// MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> onehot();
	
	//cout overloading 
	template<typename U>
	friend std::ostream& operator << (std::ostream& out, const MatrixBase<U,_Rows,1>& v);
	

	MatrixBase<T,_Rows, 1> onehot();
	int get_size();
	void push_back(T value);
	void setRandom();
	void resize(const int& _size);
	void resize(const int& _size, const T& _init);
	float norm();

};


// Genereal
template<typename T> 
class MatrixBase<T,SIZE_DYNAMIC,SIZE_DYNAMIC>  {
private:
	std::vector<std::vector<T>> matrix;
	int rows;
	int cols;

public:
	MatrixBase();
	MatrixBase(int _rows, int _cols);
	MatrixBase(int _rows, int _cols, const T& _init);
	MatrixBase(const MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& rhs);
	MatrixBase(const std::vector<std::vector<T>>& rhs);

	MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& operator=(const MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& rhs);

	virtual ~MatrixBase();


	//operator
	MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> operator+(const MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& rhs);
	MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> operator-(const MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& rhs);
	MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> operator*(const MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& rhs);
	MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& operator+=(const MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& rhs);
	MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& operator-=(const MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& rhs);
	MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& operator*=(const MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& rhs);

	MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> transpose();
	MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> square();
	MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> sqrt();
	MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> inv();

	//scalar	
	MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> operator+(const T& rhs);
	MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> operator-(const T& rhs);
	MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> operator*(const T& rhs);
	MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC> operator/(const T& rhs);
	MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& operator+=(const T& rhs);
	MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& operator-=(const T& rhs);
	MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& operator*=(const T& rhs);
	MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>& operator/=(const T& rhs);

	MatrixBase<T,SIZE_DYNAMIC,1> operator*(const MatrixBase<T,SIZE_DYNAMIC,1>& rhs);

	T& operator()(const int& row, const int& col);
	const T& operator()(const int& row, const int& col) const;
	
	

	//cout overloading 
	template<typename U>
	friend std::ostream& operator << (std::ostream& out, const MatrixBase<U,SIZE_DYNAMIC,SIZE_DYNAMIC>& m);

	int get_rows_size() const;
	int get_cols_size() const;
	void resize(int _rows, int _cols, T _init = 0);
	void setRandom();
	float norm();
};


template<typename T> 
class MatrixBase<T, SIZE_DYNAMIC,1> {
private:
	std::vector<T> vector;
	int size;

public:
	MatrixBase();
	MatrixBase(int _size, const T& _init);
	MatrixBase(const MatrixBase<T, SIZE_DYNAMIC, 1>& rhs);
	MatrixBase(const std::vector<T>& rhs);
	MatrixBase<T, SIZE_DYNAMIC, 1>& operator=(const MatrixBase<T, SIZE_DYNAMIC, 1>& rhs);
	bool operator==(const MatrixBase<T, SIZE_DYNAMIC, 1>& rhs);
	
	virtual ~MatrixBase();



	//operator
	MatrixBase<T, SIZE_DYNAMIC, 1> operator+(const MatrixBase<T, SIZE_DYNAMIC, 1>& rhs);
	MatrixBase<T, SIZE_DYNAMIC, 1> operator-(const MatrixBase<T, SIZE_DYNAMIC, 1>& rhs);
	MatrixBase<T, SIZE_DYNAMIC, 1> operator*(const MatrixBase<T, SIZE_DYNAMIC, 1>& rhs);
	MatrixBase<T, SIZE_DYNAMIC, 1>& operator+=(const MatrixBase<T, SIZE_DYNAMIC, 1>& rhs);
	MatrixBase<T, SIZE_DYNAMIC, 1>& operator-=(const MatrixBase<T, SIZE_DYNAMIC, 1>& rhs);
	MatrixBase<T, SIZE_DYNAMIC, 1>& operator*=(const MatrixBase<T, SIZE_DYNAMIC, 1>& rhs);

	//scalar
	MatrixBase<T, SIZE_DYNAMIC, 1> operator+(const T& rhs);
	MatrixBase<T, SIZE_DYNAMIC, 1> operator-(const T& rhs);
	MatrixBase<T, SIZE_DYNAMIC, 1> operator*(const T& rhs);
	MatrixBase<T, SIZE_DYNAMIC, 1> operator/(const T& rhs);
	
	MatrixBase<T, SIZE_DYNAMIC, 1>& operator+=(const T& rhs);
	MatrixBase<T, SIZE_DYNAMIC, 1>& operator-=(const T& rhs);
	MatrixBase<T, SIZE_DYNAMIC, 1>& operator*=(const T& rhs);
	MatrixBase<T, SIZE_DYNAMIC, 1>& operator/=(const T& rhs);

	//vector
	MatrixBase<T> transpose();
	T dot(const MatrixBase<T, SIZE_DYNAMIC, 1>& rhs);
	MatrixBase<T, SIZE_DYNAMIC, 1> square();
	MatrixBase<T, SIZE_DYNAMIC, 1> sqrt();
	MatrixBase<T, SIZE_DYNAMIC, 1> inv();

	T& operator()(const int& idx);
	const T& operator()(const int& idx) const;
	T& operator()(const int& begin, const int& end);

	T& operator[](const int& idx);
	const T& operator[](const int& idx) const;

	//cout overloading 
	template<typename U>
	friend std::ostream& operator << (std::ostream& out, const MatrixBase<U,SIZE_DYNAMIC,1>& v);

	MatrixBase<T, SIZE_DYNAMIC, 1> onehot();
	int get_size() const;
	void push_back(T value);
	void setRandom();
	void resize(const int& _size);
	void resize(const int& _size, const T& _init);
	float norm();
};


template<size_t _Rows,size_t _Cols,typename U> 
std::ostream& operator << (std::ostream& out, const MatrixBase<U,_Rows,_Cols>& m)
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


template<typename U>
std::ostream& operator << (std::ostream& out, const MatrixBase<U,SIZE_DYNAMIC,SIZE_DYNAMIC>& m)
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

template<typename T,size_t _Rows ,typename U> 
std::ostream& operator << (std::ostream& out, const MatrixBase<U,_Rows,1>& v)
{
	int size = v.get_size();
	out << "size : " << size << '\n';
	for (int i = 0; i < size; i++) {
		out << v(i) << '\t';
	}
	out << '\n';
	return out;
}


template<typename U>
std::ostream& operator << (std::ostream& out, const MatrixBase<U,SIZE_DYNAMIC,1>& v)
{
	int size = v.get_size();
	out << "size : " << size << '\n';
	for (int i = 0; i < size; i++) {
		out << v(i) << '\t';
	}
	out << '\n';
	return out;
}

template <typename T>
using Matrix = MatrixBase<T, SIZE_DYNAMIC, SIZE_DYNAMIC>;


#include <VanillaDNN/Math/Matrix/Matrix.cpp>
// #include "Matrix.cpp"
#endif