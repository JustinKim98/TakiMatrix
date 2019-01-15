//
// Created by jwkim98 on 19/01/14.
//

#include "../../includes/util/matrix.hpp"

namespace TakiMatrix {
    matrix::matrix(const std::vector<float>& data,
            const std::vector<size_t>& shape)
    {
        m_matrix_ptr = std::make_unique<processor::matrix_object>(data, shape);
        m_matrix_id = matrix_id++;
    }

    matrix::matrix(matrix& new_matrix)
    {
        processor::matrix_object temp = *new_matrix.m_matrix_ptr;
        m_matrix_ptr = std::make_unique<processor::matrix_object>(temp);
        m_matrix_id = matrix_id++;
    }

    matrix::matrix(matrix&& new_matrix) noexcept
    {
        m_matrix_ptr = std::move(new_matrix.m_matrix_ptr);
        m_matrix_id = matrix_id++;
    }

    size_t matrix::get_id() const { return m_matrix_id; }

    size_t matrix_hash_functor::operator()(const matrix& obj) const
    {
        return obj.get_id();
    }

} // namespace TakiMatrix