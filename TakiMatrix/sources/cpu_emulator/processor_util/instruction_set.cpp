//
// Created by jwkim98 on 19. 1. 12.
//

#include "../../../includes/cpu_emulator/processor_util/instruction_set.hpp"

namespace TakiMatrix::processor {
    isa::isa(instruction_type instruction) { this->instruction = instruction; }

    add::add(matrix_object* operand_first, matrix_object* operand_second, matrix_object* result)
            :isa(instruction_type::add)
    {
        this->operand_first = operand_first;
        this->operand_second = operand_second;
        this->result = result;
    }

    mul::mul(matrix_object* operand_first, matrix_object* operand_second, matrix_object* result)
            :isa(instruction_type::mul)
    {
        this->operand_first = operand_first;
        this->operand_second = operand_second;
        this->result = result;
    }

    dot::dot(matrix_object* operand_first, const std::function<float(float)>& functor)
            :isa(instruction_type::dot)
    {
        this->operand_first = operand_first;
        this->functor = functor;
    }

    dot::dot(matrix_object* operand_first, std::function<float(float)>&& functor)
            :isa(instruction_type::dot)
    {
        this->operand_first = operand_first;
        this->functor = std::forward<std::function<float(float)>>(functor);
    }

    transpose::transpose(matrix_object* operand_first)
            :isa(instruction_type::transpose)
    {
        this->operand_first = operand_first;
    }
} // namespace TakiMatrix::processor