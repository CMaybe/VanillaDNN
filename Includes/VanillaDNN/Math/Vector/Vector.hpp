#ifndef VANILLA_DNN_VECTOR_HPP
#define VANILLA_DNN_VECTOR_HPP

#include <VanillaDNN/Math/Matrix/Matrix.hpp>
// #include "../Matrix.hpp"
template <typename T>
using Vector = MatrixBase<T, SIZE_DYNAMIC, 1>;

template <typename T, size_t _Rows>
using VectorXd = MatrixBase<T, _Rows, 1>;

#endif