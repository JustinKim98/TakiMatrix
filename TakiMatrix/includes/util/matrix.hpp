//
// Created by jwkim98 on 19/01/14.
//

#ifndef TAKIMATRIX_MATRIX_HPP
#define TAKIMATRIX_MATRIX_HPP
#include <memory>
#include "../cpu_emulator/processor_util/matrix_object.hpp"

namespace TakiMatrix{

    size_t matrix_id = 0;

    class matrix{
    public:
        matrix(const std::vector<float>& data, const std::vector<size_t>& shape);
        matrix(matrix& new_matrix);
        matrix(matrix&& new_matrix) noexcept;

    private:
        std::unique_ptr<processor::matrix_object> m_matrix_ptr;

        size_t m_matrix_id;
    };
}

#endif //TAKIMATRIX_MATRIX_HPP
