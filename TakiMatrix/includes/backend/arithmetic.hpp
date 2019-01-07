//
// Created by jwkim98 on 19. 1. 7.
//

#ifndef TAKIMATRIX_ARTHIMETICS_HPP
#define TAKIMATRIX_ARTHIMETICS_HPP

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../util/matrix.hpp"


namespace TakiMatrix {

    Matrix& multiplyGpu(const Matrix& left, const Matrix& right);
}
#endif //TAKIMATRIX_ARTHIMETICS_HPP
