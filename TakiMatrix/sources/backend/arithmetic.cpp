//
// Created by jwkim98 on 19. 1. 7.
//

#include "../../includes/backend/arithmetic.hpp"
#include "../../includes/kernel/kernel.h"
#include "../../includes/kernel/kernelCaller.h"
#include <thread>

namespace TakiMatrix {
    void multiplyGpu(const Matrix& left, const Matrix& right, Matrix& result)
    {
        const std::vector<size_t>& leftShape = left.getShape();
        const std::vector<size_t>& rightShape = right.getShape();

        if (leftShape.at(1)!=leftShape.at(0)) {
            // TODO import Error
        }

        size_t middleSize = leftShape.at(0);

        size_t leftDataSize = left.getDataSize();
        size_t rightDataSize = right.getDataSize();

        size_t resultRow = leftShape.at(0);
        size_t resultCol = leftShape.at(1);

        size_t resultSize = resultRow*resultCol;
        size_t resultDataSize = resultSize*sizeof(float);

        const std::vector<float>& leftData = left.getData();
        const std::vector<float>& rightData = right.getData();

        const float* leftDataCpuPtr = &leftData.at(0);
        const float* rightDataCpuPtr = &rightData.at(0);
        auto* resultDataCpuPtr = (float*) malloc(resultDataSize);

        callMultiply(leftDataCpuPtr, rightDataCpuPtr, resultDataCpuPtr, resultRow,
                resultCol, middleSize);

        result.assignData(resultDataCpuPtr, resultSize,
                std::vector<size_t>{resultRow, resultCol});

        free(resultDataCpuPtr);
    }
} // namespace TakiMatrix
