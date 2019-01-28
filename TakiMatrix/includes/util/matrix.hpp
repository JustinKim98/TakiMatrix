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
        /**
         * constructor for matrix object
         * @param processor : inherited processor for processing the thread
         * @param data : data to initialize the matrix
         * @param shape : shape of the data (shape must match the size of data)
         */
        matrix(std::reference_wrapper<process> processor, const std::vector<float>& data,
                const std::vector<size_t>& shape);
        /**
         * construct matrix object using matrix_object_ptr
         * @param processor : processor for processing the thread
         * @param matrix_object_ptr : shared pointer to matrix_object_ptr to
         * initialize the matrix
         */
        explicit matrix(std::reference_wrapper<process> processor,
                std::shared_ptr<matrix_object>& matrix_object_ptr);

        /**
         * copy constructor for this matrix
         *  constructs new matrix and shares ownership of matrix_object with
         * 'first'
         * @param new_matrix : matrix object to copy
         */
        matrix(matrix& new_matrix) = default;

        matrix(matrix&& new_matrix) = default;
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
         * @param first : matrix to compare with
         * @return : true if matrices are identical
         */
        bool operator==(const matrix& first);

        bool operator!=(const matrix& first);

    private:
        /// pointer to matrix_object
        std::shared_ptr<matrix_object> m_matrix_ptr;
        std::reference_wrapper<process> m_processor;
    };
} // namespace TakiMatrix

#endif // TAKIMATRIX_MATRIX_HPP
