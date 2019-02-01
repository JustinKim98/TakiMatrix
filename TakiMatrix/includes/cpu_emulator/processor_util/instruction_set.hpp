/**
 * @file : instruction_set.hpp
 * @author : Justin Kim
 * @version : 1.0
 * basic instruction set for processor
 * new instructions may be added or instructions may be refactored after this
 * contains shared_ptr for matrix_objects for processing
 */

#ifndef TAKIMATRIX_INSTRUCTION_SET_HPP
#define TAKIMATRIX_INSTRUCTION_SET_HPP

#include "matrix_object.hpp"
#include <functional>
#include <memory>

namespace TakiMatrix::processor {
    enum class instruction_type {
        add,
        sub,
        mul,
        dot,
        transpose,
    };

    class instruction {
    public:
        /**
         * constructs instruction without functor
         *
         * @param type : type of this operation
         * @param operand_first : ptr for first operand to calculate
         * @param operand_second : ptr for second operand to calculate
         * @param result : ptr to store result
         */
        explicit instruction(enum instruction_type type,
                const std::shared_ptr<matrix_object>& operand_first,
                const std::shared_ptr<matrix_object>& operand_second,
                std::shared_ptr<matrix_object>& result);

        /**
         * constructs instruction with functor
         * functor object can be used for dot operation(must have __device__
         * declaration)
         *
         * @param type
         * @param operand_first
         * @param operand_second
         * @param result
         * @param functor
         */
        explicit instruction(enum instruction_type type,
                const std::shared_ptr<matrix_object>& operand_first,
                const std::shared_ptr<matrix_object>& operand_second,
                std::shared_ptr<matrix_object>& result,
                const std::function<float(float)>& functor);
        /**
         * gets shared_ptr for result of this instruction
         * @return : ptr to the result
         */
        std::shared_ptr<matrix_object> result_ptr();

        std::shared_ptr<matrix_object> first_operand_ptr();

        std::shared_ptr<matrix_object> second_operand_ptr();

    protected:
        /// type of this instruction
        instruction_type m_instruction_type;
        /// first operand
        const std::shared_ptr<matrix_object> m_operand_first;
        /// second operand
        const std::shared_ptr<matrix_object> m_operand_second;
        /// result
        std::shared_ptr<matrix_object> m_result;
        /// functor for dot operation
        const std::function<float(float)> m_functor = nullptr;
    };
} // namespace TakiMatrix::processor
#endif // TAKIMATRIX_INSTRUCTION_SET_HPP
