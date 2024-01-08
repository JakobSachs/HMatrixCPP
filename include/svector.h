#ifndef SVECTOR_H
#define SVECTOR_H

#include <array>

#ifdef __APPLE__
#include <cblas_new.h>
#else
#include <cblas.h>
#endif

#include <cstddef>
#include <type_traits>

namespace HMatrix {

// SVector is a vector of size N that is statically allocated.
template <typename T, std::size_t N> class SVector {
private:
  std::array<T, N> buff;

public:
  // Default constructor (TODO: initalize from other container)
  SVector() : buff() {}

  // Overloaded subscript operator for non-const access
  // Throws std::out_of_range if index is out of bounds
  T &operator[](std::size_t i) {
    if (i >= N) {
      throw std::out_of_range("SVector index out of range");
    }
    return buff[i];
  }

  // Const subscript operator
  const T &operator[](std::size_t i) const {
    if (i >= N) {
      throw std::out_of_range("SVector index out of range");
    }
    return buff[i];
  }

  // Returns the size of the vector (number of elements)
  std::size_t size() const { return N; }

  // Iterator class
  class iterator {
  public:
    iterator &operator++() { // Prefix increment
      ptr++;
      return *this;
    }
    iterator operator++(int) { // Postfix increment
      iterator temp = *this;
      ++(*this);
      return temp;
    }
    // Comparison operators
    bool operator==(const iterator &other) const { return ptr == other.ptr; }
    bool operator!=(const iterator &other) const { return ptr != other.ptr; }

    // Dereference operator
    T &operator*() const { return *ptr; }

  private:
    T *ptr;
  };

  // iterator functions
  iterator begin() { return iterator(buff.data()); }
  iterator end() { return iterator(buff.data() + N); }

  // Simple arithmetic operations

  // Vector addition - adds corresponding elements of two vectors
  SVector<T, N> operator+(const SVector<T, N> &other) const {
    SVector<T, N> result;

    for (std::size_t i = 0; i < N; i++) {
      result[i] = buff[i] + other[i];
    }
    return result;
  }

  // Vector subtraction - subtracts corresponding elements of two vectors
  SVector<T, N> operator-(const SVector<T, N> &other) const {
    return this + (other * -1); // this is probably dumb but ya know
  }

  // Scalar multiplication - multiplies each element of the vector by a scalar
  SVector<T, N> operator*(const T &scalar) const {
    SVector<T, N> result = this;
    for (std::size_t i = 0; i < N; i++) {
      result[i] *= scalar;
    }
    return result;
  }

  // Scalar division - divides each element of the vector by a scalar
  // Throws std::domain_error if scalar is 0
  SVector<T, N> operator/(const T &scalar) const {
    if (scalar == 0) {
      throw std::domain_error("SVector division by zero");
    }

    SVector<T, N> result = this;
    for (std::size_t i = 0; i < N; i++) {
      result[i] /= scalar;
    }
    return result;
  }

  // Comparison operator
  bool operator==(const SVector<T, N> &other) const {
    for (std::size_t i = 0; i < N; i++) {
      if (buff[i] != other[i]) {
        return false;
      }
    }
    return true;
  }

  // Dot product (TODO: allow other types of vectors)

  // Generic Dot product for non-float/double vectors
  template <typename U = T>
  typename std::enable_if<!std::is_same<U, float>::value &&
                              !std::is_same<U, double>::value,
                          T>::type
  dot(const SVector<T, N> &other) const {
    T sum = T();
    for (std::size_t i = 0; i < N; ++i) {
      sum += buff[i] * other[i];
    }
    return sum;
  }

  // Specialized dot product for float
  template <typename U = T>
  typename std::enable_if<std::is_same<U, float>::value, float>::type
  dot(const SVector<float, N> &other) const {
    return cblas_sdot(N, buff.data(), 1, other.buff.data(), 1);
  }

  // Specialized dot product for double
  template <typename U = T>
  typename std::enable_if<std::is_same<U, double>::value, double>::type
  dot(const SVector<double, N> &other) const {
    return cblas_ddot(N, buff.data(), 1, other.buff.data(), 1);
  }
};
} // namespace HMatrix

#endif
