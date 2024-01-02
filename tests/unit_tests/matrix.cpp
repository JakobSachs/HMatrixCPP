#include "matrix.h"
#include "gtest/gtest.h"

TEST(HierarchicalMatrixTest, Constructor) {
  Matrix matrix;
  EXPECT_EQ(matrix.get_dimension(), 0);
}
