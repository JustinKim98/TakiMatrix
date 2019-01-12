//
// Created by jwkim98 on 19. 1. 12.
//

#ifndef TAKIMATRIX_INSTRUCTION_SET_HPP
#define TAKIMATRIX_INSTRUCTION_SET_HPP

#include "../../../includes/util/matrix.hpp"

namespace TakiMatrix {
    namespace processor {
        enum class instruction_type {
            add,
            sub,
            mul,
            malloc_gpu,
            malloc_cpu,
            copy_d2h,
            copy_h2d,
            free_cpu,
            free_gpu,
            transpose,
        };

        class isa {
        public:
            instruction_type instruction;

        protected:
            isa(instruction_type instruction);
        };

        class add_inst : isa {
            add_inst(matrix* operand_first, matrix* operand_second, matrix* result)
                    :isa(instruction_type::add)
            {
                this->operand_first = operand_first;
                this->operand_second = operand_second;
                this->result = result;
            }

        private:
            const matrix* operand_first;
            const matrix* operand_second;
            matrix* result;
        };
    } // namespace processor
} // namespace TakiMatrix

#endif // TAKIMATRIX_INSTRUCTION_SET_HPP
