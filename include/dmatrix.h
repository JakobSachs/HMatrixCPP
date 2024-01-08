#ifndef DMATRIX_H
#define DMATRIX_H

#include <array>

#ifdef __APPLE__
#include <cblas_new.h>
#else
#include <cblas.h>
#endif

#include <cstddef>
#include <type_traits>

namespace HMatrix {

// DMatrix is a dense matrix of size M,Nthat is statically allocated.
template <typename T, std::size_t M, std::size_t N> class DMatrix {
private:
  std::array<T, M * N> buff;

public:
  // Default constructor (TODO: initalize from other container)
  DMatrix() : buff() {}

  // Subscript operator for non-const access
  T &operator()(std::size_t i, std::size_t j) {
    if (i >= M || j >= N) {
      throw std::out_of_range("DMatrix index out of range");
    }
    return buff[i * N + j];
  }

  // Const subscript operator
  T operator()(std::size_t i, std::size_t j) const {
    if (i >= M || j >= N) {
      throw std::out_of_range("DMatrix index out of range");
    }
    return buff[i * N + j];
  }

  // Indexing operator for non-const access
  T &operator[](std::size_t i) {
    if (i >= M * N) {
      throw std::out_of_range("DMatrix index out of range");
    }
    return buff[i];
  }

  // Returns the number of rows
  std::size_t rows() const { return M; }

  // Returns the number of columns
  std::size_t cols() const { return N; }
};

} // namespace HMatrix

#endif // DMATRIX_H
