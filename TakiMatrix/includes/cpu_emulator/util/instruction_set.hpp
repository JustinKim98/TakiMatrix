//
// Created by jwkim98 on 19. 1. 12.
//

#ifndef TAKIMATRIX_INSTRUCTION_SET_HPP
#define TAKIMATRIX_INSTRUCTION_SET_HPP

#include "../../../includes/util/matrix.hpp"
#include <functional>

namespace TakiMatrix {
    namespace processor {
        enum class instruction_type {
            add,
            sub,
            mul,
            dot,
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
        protected:
            isa(instruction_type instruction);

            instruction_type instruction;
        };

        class add : isa {
        public:
            add(matrix* operand_first, matrix* operand_second, matrix* result);

        private:
            const matrix* operand_first;
            const matrix* operand_second;
            matrix* result;
        };

        class sub : isa {
        public:
            sub(matrix* operand_first, matrix* operand_second, matrix* result);

        private:
            const matrix* operand_first;
            const matrix* operand_second;
            matrix* result;
        };

        class mul : isa {
        public:
            mul(matrix* operand_first, matrix* operand_second, matrix* result);

        private:
            const matrix* operand_first;
            const matrix* operand_second;
            matrix* result;
        };

        class dot : isa {
        public:
            dot(matrix* operand_first, const std::function<float(float)>& functor);

            dot(matrix* operand_first, std::function<float(float)>&& functor);

        private:
            const matrix* operand_first;
            std::function<float(float)> functor;
        };

    } // namespace processor
} // namespace TakiMatrix

#endif // TAKIMATRIX_INSTRUCTION_SET_HPP
