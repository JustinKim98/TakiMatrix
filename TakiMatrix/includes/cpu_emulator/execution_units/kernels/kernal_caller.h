//
// Created by jwkim98 on 19/02/03.
//

#ifndef TAKIMATRIX_KERNAL_CALLER_H
#define TAKIMATRIX_KERNAL_CALLER_H

#include "kernels.h"
#include <cstdio>

namespace kernel {
    /**
     * calls add kernel
     *
     * @param operand_a : pointer to data of first operand
     * @param operand_b : pointer to data of second operand
     * @param size : element size of operand_a and operand_b
     */
    void call_add(float* operand_a, float* operand_b, size_t size);
    /**
     * calls sub kernel
     *
     * @param operand_a
     * @param operand_b
     * @param size
     */
    void call_sub(float* operand_a, float* operand_b, size_t size);
    /**
     * calls mul kernel
     *
     * @param operand_a : pointer to data of first operand
     * @param operand_b : pointer to data of second operand
     * @param result : pointer to data of the result
     * @param middle_count : size of operand_a's column and operand_b's row
     * @param first_row_count : number of rows of operand_a
     * @param second_col_count : number of cols of operand_b
     * @param dimension_count : number of dimensions of operand_a and operand_b
     */
    void call_mul(float* operand_a, float* operand_b, float* result,
            size_t middle_count, size_t first_row_count,
            size_t second_col_count, size_t dimension_count);

    /**
     * calls user-customized function on the device
     * @tparam Func : float(float)
     * @param operand_a : pointer to data of first operand
     * @param func : function of type float(float) to be applied to every element
     * @param size : element size of operand_a;
     */
    template<typename Func>
    void call_dot(float* operand_a, Func func, size_t size);

} // namespace kernel
#endif // TAKIMATRIX_KERNAL_CALLER_H
