#include "dmatrix.h"
#include "gtest/gtest.h"

using namespace HMatrix;

// Test constructor
TEST(DMatrixTest, Constructor) {
  DMatrix<int, 2, 3> m;
  ASSERT_EQ(m.rows(), 2);
  ASSERT_EQ(m.cols(), 3);
}

TEST(DMatrixTest, AssignmentOperator) {
  DMatrix<int, 2, 3> m;
  m(0, 0) = 1;
  m(0, 1) = 2;
  m(0, 2) = 3;
  m(1, 0) = 1;
  m(1, 1) = 2;
  m(1, 2) = 3;

  ASSERT_EQ(m.rows(), 2);
  ASSERT_EQ(m.cols(), 3);

  ASSERT_EQ(m(0, 0), 1);
  ASSERT_EQ(m(0, 1), 2);
  ASSERT_EQ(m(0, 2), 3);
  ASSERT_EQ(m(1, 0), 1);
  ASSERT_EQ(m(1, 1), 2);
  ASSERT_EQ(m(1, 2), 3);
}

TEST(DMatrixTest, BoundChecking) {
  DMatrix<int, 2, 3> m;
  EXPECT_THROW({ m(2, 0) = 1; }, std::out_of_range);
  EXPECT_THROW({ m(0, 3) = 1; }, std::out_of_range);

  EXPECT_THROW({ m[2 * 3] = 1; }, std::out_of_range);
  EXPECT_THROW({ m[-1] = 1; }, std::out_of_range);
}

TEST(DMatrixTest, RowSpan) {
  DMatrix<int, 2, 3> m;
  m(0, 0) = 1;
  m(0, 1) = 2;
  m(0, 2) = 3;
  m(1, 0) = 1;
  m(1, 1) = 2;
  m(1, 2) = 3;

  auto row = m.row(0);
  ASSERT_EQ(row.size(), 3);
  ASSERT_EQ(row[0], 1);
  ASSERT_EQ(row[1], 2);
  ASSERT_EQ(row[2], 3);
}

TEST(DMatrixTest, ColSpan) {
  DMatrix<int, 2, 3> m;
  m(0, 0) = 1;
  m(0, 1) = 2;
  m(0, 2) = 3;
  m(1, 0) = 4;
  m(1, 1) = 5;
  m(1, 2) = 6;

  auto col = m.col(0);
  ASSERT_EQ(col.size(), 2);
  ASSERT_EQ(col[0], 1);
  ASSERT_EQ(col[1], 4);

  col = m.col(1);
  ASSERT_EQ(col.size(), 2);
  ASSERT_EQ(col[0], 2);
  ASSERT_EQ(col[1], 5);

  col = m.col(2);
  ASSERT_EQ(col.size(), 2);
  ASSERT_EQ(col[0], 3);
  ASSERT_EQ(col[1], 6);
}

TEST(DMatrixTest, Add) {
  DMatrix<int, 2, 3> m;
  m(0, 0) = 1;
  m(0, 1) = 2;
  m(0, 2) = 3;
  m(1, 0) = 4;
  m(1, 1) = 5;
  m(1, 2) = 6;
  DMatrix<int, 2, 3> n;
  n(0, 0) = 1;
  n(0, 1) = 2;
  n(0, 2) = 3;
  n(1, 0) = 4;
  n(1, 1) = 5;
  n(1, 2) = 6;

  auto sum = m + n;
  ASSERT_EQ(sum.rows(), 2);
  ASSERT_EQ(sum.cols(), 3);
  ASSERT_EQ(sum(0, 0), 2);
  ASSERT_EQ(sum(0, 1), 4);
  ASSERT_EQ(sum(0, 2), 6);
  ASSERT_EQ(sum(1, 0), 8);
  ASSERT_EQ(sum(1, 1), 10);
  ASSERT_EQ(sum(1, 2), 12);
}

TEST(DMatrixTest, ScalarMultiply) {
  DMatrix<int, 2, 3> m;
  m(0, 0) = 1;
  m(0, 1) = 2;
  m(0, 2) = 3;
  m(1, 0) = 4;
  m(1, 1) = 5;
  m(1, 2) = 6;

  auto prod = m * 2;
  ASSERT_EQ(prod.rows(), 2);
  ASSERT_EQ(prod.cols(), 3);
  ASSERT_EQ(prod(0, 0), 2);
  ASSERT_EQ(prod(0, 1), 4);
  ASSERT_EQ(prod(1, 2), 12);
}

TEST(DMatrixTest, MatrixMulti) {
  DMatrix<int, 2, 3> m;
  m(0, 0) = 1;
  m(0, 1) = 2;
  m(0, 2) = 3;
  m(1, 0) = 4;
  m(1, 1) = 5;
  m(1, 2) = 6;

  DMatrix<int, 3, 2> n;
  n(0, 0) = 1;
  n(0, 1) = 2;
  n(1, 0) = 3;
  n(1, 1) = 4;
  n(2, 0) = 5;
  n(2, 1) = 6;

  auto prod = m * n;
  ASSERT_EQ(prod.rows(), 2);
  ASSERT_EQ(prod.cols(), 2);
  ASSERT_EQ(prod(0, 0), 22);
  ASSERT_EQ(prod(0, 1), 28);
  ASSERT_EQ(prod(1, 0), 49);
  ASSERT_EQ(prod(1, 1), 64);
}

TEST(DMatrixTest, MatrixMultiFloat) {
  DMatrix<float, 2, 3> m;
  m(0, 0) = 1.0;
  m(0, 1) = 2.0;
  m(0, 2) = 3.0;
  m(1, 0) = 4.0;
  m(1, 1) = 5.0;
  m(1, 2) = 6.0;

  DMatrix<float, 3, 2> n;
  n(0, 0) = 1.0;
  n(0, 1) = 2.0;
  n(1, 0) = 3.0;
  n(1, 1) = 4.0;
  n(2, 0) = 5.0;
  n(2, 1) = 6.0;

  auto prod = m * n;
  ASSERT_EQ(prod.rows(), 2);
  ASSERT_EQ(prod.cols(), 2);
  ASSERT_EQ(prod(0, 0), 22.0);
  ASSERT_EQ(prod(0, 1), 28.0);
  ASSERT_EQ(prod(1, 0), 49.0);
  ASSERT_EQ(prod(1, 1), 64.0);
}

TEST(DMatrixTest, MatrixVectorTest) {
  DMatrix<int, 2, 2> m;
  m(1, 0) = 1;
  m(0, 1) = 1;

  SVector<int, 2> v;
  v[0] = 1;
  v[1] = 2;

  auto prod = m * v;
  ASSERT_EQ(prod.size(), 2);
  ASSERT_EQ(prod[0], 2);
  ASSERT_EQ(prod[1], 1);
}

TEST(DMatrixTest, MatrixVectorTestFloat) {
  DMatrix<float, 2, 2> m;
  m(1, 0) = 1.0;
  m(0, 1) = 1.0;

  SVector<float, 2> v;
  v[0] = 1.0;
  v[1] = 2.0;

  auto prod = m * v;
  ASSERT_EQ(prod.size(), 2);
  ASSERT_EQ(prod[0], 2);
  ASSERT_EQ(prod[1], 1);
}
