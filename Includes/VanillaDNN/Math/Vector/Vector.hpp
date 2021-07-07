#ifndef VANILLA_DNN_VECTOR_HPP
#define VANILLA_DNN_VECTOR_HPP

#include <VanillaDNN/Math/Matrix/Matrix.hpp>

template <typename T, size_t Rows>
using Vector = Matrix<T, Rows, 1>;


#endif