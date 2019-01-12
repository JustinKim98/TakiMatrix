//
// Created by jwkim98 on 19. 1. 12.
//

#include "../../../includes/cpu_emulator/util/instruction_set.hpp"

namespace TakiMatrix {
    namespace processor {
        isa::isa(instruction_type instruction) { this->instruction = instruction; }

        add::add(matrix* operand_first, matrix* operand_second,
                matrix* result)
                :isa(instruction_type::add)
        {
            this->operand_first = operand_first;
            this->operand_second = operand_second;
            this->result = result;
        }

        sub::sub(matrix* operand_first, matrix* operand_second,
                matrix* result)
                :isa(instruction_type::sub)
        {
            this->operand_first = operand_first;
            this->operand_second = operand_second;
            this->result = result;
        }

        mul::mul(matrix* operand_first, matrix* operand_second,
                matrix* result)
                :isa(instruction_type::mul)
        {
            this->operand_first = operand_first;
            this->operand_second = operand_second;
            this->result = result;
        }

        dot::dot(matrix* operand_first,
                const std::function<float(float)>& functor)
                :isa(instruction_type::dot)
        {
            this->operand_first = operand_first;
            this->functor = functor;
        }

        dot::dot(matrix* operand_first, std::function<float(float)>&& functor)
                :isa(instruction_type::dot)
        {
            this->operand_first = operand_first;
            this->functor = std::forward<std::function<float(float)>>(functor);
        }

    } // namespace processor
} // namespace TakiMatrix