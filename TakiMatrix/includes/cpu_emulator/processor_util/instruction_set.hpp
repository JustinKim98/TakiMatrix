//
// Created by jwkim98 on 19. 1. 12.
// takion -> 빠르다/
//

#ifndef TAKIMATRIX_INSTRUCTION_SET_HPP
#define TAKIMATRIX_INSTRUCTION_SET_HPP

#include "matrix_object.hpp"
#include <functional>

namespace TakiMatrix::processor {
    enum class instruction_type {
        add,
        mul,
        dot,
        transpose,
    };

    class isa {
    public:
        isa(const isa& instruction);

        virtual ~isa() = default;

    protected:
        isa(instruction_type instruction);

        instruction_type instruction;
    };


    class add : isa {
    public:
        add(matrix_object* operand_first, matrix_object* operand_second, matrix_object* result);

    private:
        const matrix_object* operand_first;
        const matrix_object* operand_second;
        matrix_object* result;
    };

    class mul : isa {
    public:
        mul(matrix_object* operand_first, matrix_object* operand_second, matrix_object* result);

    private:
        const matrix_object* operand_first;
        const matrix_object* operand_second;
        matrix_object* result;
    };

    class dot : isa {
    public:
        dot(matrix_object* operand_first, const std::function<float(float)>& functor);

        dot(matrix_object* operand_first, std::function<float(float)>&& functor);

    private:
        const matrix_object* operand_first;
        std::function<float(float)> functor;
    };

    class transpose : isa {
    public:
        explicit transpose(matrix_object* operand_first);

    private:
        const matrix_object* operand_first;
    };

} // namespace TakiMatrix::processor

#endif // TAKIMATRIX_INSTRUCTION_SET_HPP
