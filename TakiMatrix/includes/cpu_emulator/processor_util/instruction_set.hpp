/**
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
        explicit instruction(enum instruction_type type,
                const std::shared_ptr<matrix_object>& operand_first,
                const std::shared_ptr<matrix_object>& operand_second,
                std::shared_ptr<matrix_object>& result);

        explicit instruction(enum instruction_type type,
                const std::shared_ptr<matrix_object>& operand_first,
                const std::shared_ptr<matrix_object>& operand_second,
                std::shared_ptr<matrix_object>& result,
                const std::function<float(float)>& functor);

        virtual ~instruction() = default;

        std::shared_ptr<matrix_object> get_result_ptr();

    protected:
        instruction_type m_instruction_type;
        const std::shared_ptr<matrix_object> m_operand_first;
        const std::shared_ptr<matrix_object> m_operand_second;
        std::shared_ptr<matrix_object> m_result;
        const std::function<float(float)> m_functor = nullptr;
    };
} // namespace TakiMatrix::processor
#endif // TAKIMATRIX_INSTRUCTION_SET_HPP
