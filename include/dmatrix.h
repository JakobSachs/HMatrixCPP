#ifndef DMATRIX_H
#define DMATRIX_H

#include "svector.h"
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

  // Public getter for const access to buff
  const T *data() const { return buff.data(); }

  // Public getter for non-const access to buff
  T *data() { return buff.data(); }

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

  // Indexing operator for non-const access (row-major order)
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

  // Class for returning rows/cols as span
  class span {

  private:
    T *ptr;
    std::size_t _size, stride_size;

    template <typename U>
    using GenericType =
        typename std::enable_if<std::is_same<U, T>::value ||
                                std::is_same<U, const T>::value>::type;

  public:
    span(T *ptr, std::size_t size, std::size_t stride_size)
        : ptr(ptr), _size(size), stride_size(stride_size) {}
    T *begin() const { return ptr; }
    T *end() const { return ptr + _size * stride_size; }
    std::size_t size() const { return _size; }
    T &operator[](std::size_t i) {
      if (i >= _size) {
        throw std::out_of_range("DMatrix index out of range");
      }
      return ptr[i * stride_size];
    }
  };

  // Row span
  span row(std::size_t i) {
    if (i >= M) {
      throw std::out_of_range("DMatrix index out of range");
    }
    return span(&buff[i * N], N, 1);
  }

  // Const row span
  const span row(std::size_t i) const {
    if (i >= M) {
      throw std::out_of_range("DMatrix index out of range");
    }
    return span(&buff[i * N], N, 1);
  }

  // Column span
  span col(std::size_t j) {
    if (j >= N) {
      throw std::out_of_range("DMatrix index out of range");
    }
    return span(&buff[j], M, N);
  }

  // Const column span
  const span col(std::size_t j) const {
    if (j >= N) {
      throw std::out_of_range("DMatrix index out of range");
    }
    return span(&buff[j], M, N);
  }

  // TODO: Maybe add diagonal span?

  // Basic arithmetic operators
  DMatrix<T, M, N> operator+(const DMatrix<T, M, N> &rhs) {
    auto res = *this;
    for (std::size_t i = 0; i < M * N; ++i) {
      res.buff[i] += rhs.buff[i];
    }
    return res;
  }

  DMatrix<T, M, N> operator*(const T &rhs) {
    auto res = *this;
    for (std::size_t i = 0; i < M * N; ++i) {
      res.buff[i] *= rhs;
    }
    return res;
  }

  // Generic Matrix multiplication for non-floats (this (MxN) * other(NxK))
  template <typename U = T, std::size_t K>
  typename std::enable_if<!std::is_same<U, float>::value &&
                              !std::is_same<U, double>::value,
                          DMatrix<T, M, K>>::type
  operator*(const DMatrix<T, N, K> &other) const {
    DMatrix<T, M, K> result;

    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = 0; j < K; ++j) {
        result(i, j) = 0;
        for (std::size_t k = 0; k < N; ++k) {
          result(i, j) += (*this)(i, k) * other(k, j);
        }
      }
    }

    return result;
  }

  // specialization for floats & doubles
  template <typename U = T, std::size_t K>
  typename std::enable_if<std::is_same<U, float>::value, DMatrix<T, M, K>>::type
  operator*(const DMatrix<float, N, K> &other) const {
    DMatrix<T, M, K> result;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, K, N, 1.0f,
                buff.data(), N, other.data(), K, 0.0f, result.data(), K);

    return result;
  }

  template <typename U = T, std::size_t K>
  typename std::enable_if<std::is_same<U, double>::value,
                          DMatrix<T, M, K>>::type
  operator*(const DMatrix<double, N, K> &other) const {
    DMatrix<T, M, K> result;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, K, N, 1.0f,
                buff.data(), N, other.data(), K, 0.0f, result.data(), K);

    return result;
  }

  // Generic Vector multiplication for non-floats (this (MxN) *(M))
  template <typename U = T>
  typename std::enable_if<!std::is_same<U, float>::value &&
                              !std::is_same<U, double>::value,
                          SVector<U, N>>::type
  operator*(const SVector<U, M> &other) const {
    SVector<T, N> result;
    for (std::size_t i = 0; i < N; ++i) {
      result[i] = 0;
      for (std::size_t j = 0; j < M; ++j) {
        result[i] += (*this)(j, i) * other[j];
      }
    }
    return result;
  }

  // specialization for floats & doubles
  template <typename U = T>
  typename std::enable_if<std::is_same<U, float>::value, SVector<U, N>>::type
  operator*(const SVector<float, M> &other) const {
    SVector<T, N> result;

    cblas_sgemv(CblasRowMajor, CblasNoTrans, M, N, 1.0f, buff.data(), N,
                other.data(), 1, 0.0f, result.data(), 1);

    return result;
  }

  template <typename U = T>
  typename std::enable_if<std::is_same<U, double>::value, SVector<U, N>>::type
  operator*(const SVector<double, M> &other) const {
    SVector<T, N> result;

    cblas_dgemv(CblasRowMajor, CblasNoTrans, M, N, 1.0f, buff.data(), N,
                other.data(), 1, 0.0f, result.data(), 1);

    return result;
  }
};

} // namespace HMatrix

#endif // DMATRIX_H
