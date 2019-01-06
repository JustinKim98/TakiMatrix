#include "SimpleTest.hpp"
#include "../../includes/test.h"
#include <gtest/gtest.h>
#include <iostream>

TEST(SimpleTest, Add)
{
    EXPECT_EQ(5, Add(2, 3));
    std::cout << "called Test" << std::endl;
}

TEST(AddKernel, add_with_cuda)
{
    add_with_cuda();
    std::cout << "add with cuda" << std::endl;
}