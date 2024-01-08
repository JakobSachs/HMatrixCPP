#include "svector.h"
#include "gtest/gtest.h"

TEST(SVectorTest, Constructor) {
  HMatrix::SVector<double, 3> vector;
  EXPECT_EQ(vector.size(), 3);
  EXPECT_EQ(vector[2], 0.0);

  HMatrix::SVector<double, 0> empty_vector;
  EXPECT_EQ(empty_vector.size(), 0.0);
}

TEST(SVectorTest, AssignmentOperator) {
  HMatrix::SVector<double, 3> vector;
  vector[0] = 1.0;
  vector[1] = 2.0;
  vector[2] = 3.0;
  HMatrix::SVector<double, 3> copy_vector;
  copy_vector = vector;
  EXPECT_EQ(copy_vector.size(), 3);
  EXPECT_EQ(copy_vector[0], 1.0);
  EXPECT_EQ(copy_vector[1], 2.0);
  EXPECT_EQ(copy_vector[2], 3.0);
}

TEST(SVectorTest, DotProduct) {
  // int test
  HMatrix::SVector<int, 3> i_vector;
  i_vector[0] = 1;
  i_vector[1] = 2;
  i_vector[2] = 3;
  HMatrix::SVector<int, 3> i_other_vector;
  i_other_vector[0] = 4;
  i_other_vector[1] = 5;
  i_other_vector[2] = 6;
  EXPECT_EQ(i_vector.dot(i_other_vector), 32);

  // float test
  HMatrix::SVector<float, 3> f_vector;
  f_vector[0] = 1.0;
  f_vector[1] = 2.0;
  f_vector[2] = 3.0;
  HMatrix::SVector<float, 3> f_other_vector;
  f_other_vector[0] = 4.0;
  f_other_vector[1] = 5.0;
  f_other_vector[2] = 6.0;
  EXPECT_EQ(f_vector.dot(f_other_vector), 32.0);
  // double test
  HMatrix::SVector<double, 3> d_vector;
  d_vector[0] = 1.0;
  d_vector[1] = 2.0;
  d_vector[2] = 3.0;
  HMatrix::SVector<double, 3> d_other_vector;
  d_other_vector[0] = 4.0;
  d_other_vector[1] = 5.0;
  d_other_vector[2] = 6.0;
  EXPECT_EQ(d_vector.dot(d_other_vector), 32.0);
}
