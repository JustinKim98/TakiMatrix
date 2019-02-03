/**
 * @file : instruction_set.cpp
 */

#include "../../../includes/cpu_emulator/processor_util/instruction_set.hpp"

namespace TakiMatrix::processor {
    instruction::instruction(enum instruction_type type,
            const std::shared_ptr<matrix_object>& operand_first,
            const std::shared_ptr<matrix_object>& operand_second,
            std::shared_ptr<matrix_object>& result)
            :m_instruction_type(type), m_operand_first(operand_first),
             m_operand_second(operand_second), m_result(result) { }

    instruction::instruction(enum instruction_type type,
            const std::shared_ptr<matrix_object>& operand_first,
            const std::shared_ptr<matrix_object>& operand_second,
            std::shared_ptr<matrix_object>& result,
            const std::function<float(float)>& functor)
            :m_instruction_type(type), m_operand_first(operand_first),
             m_operand_second(operand_second), m_result(result), m_functor(functor) { }

    std::shared_ptr<matrix_object> instruction::result_ptr() { return m_result; }

    std::shared_ptr<matrix_object> instruction::first_operand_ptr()
    {
        return m_operand_first;
    }

    std::shared_ptr<matrix_object> instruction::second_operand_ptr()
    {
        return m_operand_second;
    }

    const std::function<float(float)> instruction::functor() { return m_functor; }

} // namespace TakiMatrix::processor