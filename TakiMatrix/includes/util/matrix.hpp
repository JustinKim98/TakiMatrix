//
// Created by jwkim98 on 19/01/14.
//

#ifndef TAKIMATRIX_MATRIX_HPP
#define TAKIMATRIX_MATRIX_HPP

#include "../cpu_emulator/processor_util/instruction_set.hpp"
#include "../cpu_emulator/processor_util/matrix_object.hpp"
#include "../cpu_emulator/system_agent/process.hpp"
#include <condition_variable>
#include <memory>

using namespace TakiMatrix::processor;

namespace TakiMatrix {

    size_t matrix_id = 0;

    class matrix {
    public:
        matrix(process& processor,const std::vector<float>& data, const std::vector<size_t>& shape);

        explicit matrix(process& processor, matrix_object* matrix_object_ptr);

        matrix(matrix& new_matrix);

        matrix(matrix&& new_matrix) noexcept;

        ~matrix();

        /**
         * @brief : decodes following operation to ISA and renames matrix_object value
         * matrix_object's data is updated when corresponding operation is committed
         * @param first : matrix to perform addition(must have same shape)
         * @return : matrix as a result (they don't necessarily contain result after
         * this operation)
         */
        matrix operator+(const matrix& first);

        matrix operator-(const matrix& first);

        matrix operator*(const matrix& first);
        /**
         * @brief : compares matrices element by element
         * these instructions does not create isa
         * @param first : matrix to compare with
         * @return : true if matrices are identical
         */
        bool operator==(const matrix& first);

        bool operator!=(const matrix& first);

        /// think about branches
        size_t get_id() const;

    private:
        /// pointer to matrix_object
        /// deleted by destructor only when matrix is l-value reference
        matrix_object* m_matrix_ptr;

        size_t m_matrix_id;;

        std::reference_wrapper<process> m_processor;
    };

    struct matrix_hash_functor {
        size_t operator()(const matrix& obj) const;
    };
} // namespace TakiMatrix

#endif // TAKIMATRIX_MATRIX_HPP
