//
// Created by jwkim98 on 19/01/14.
//

#include "../../includes/util/matrix.hpp"

namespace TakiMatrix {
    matrix::matrix(const std::vector<float>& data,
            const std::vector<size_t>& shape)
    {
        m_matrix_ptr = new matrix_object(data, shape);
        m_matrix_id = matrix_id++;
    }

    matrix::matrix(matrix_object* matrix_object_ptr)
    {
        m_matrix_ptr = matrix_object_ptr;
        m_matrix_id = matrix_id++;
    }

    matrix::matrix(matrix& new_matrix)
    {
        processor::matrix_object temp = *new_matrix.m_matrix_ptr;
        m_matrix_ptr = new matrix_object(temp);
        m_matrix_id = matrix_id++;
    }

    matrix::matrix(matrix&& new_matrix) noexcept
    {
        m_matrix_ptr = new_matrix.m_matrix_ptr;
        m_matrix_id = matrix_id++;
    }

    matrix::~matrix(){
        delete m_matrix_ptr;
    }

    size_t matrix::get_id() const { return m_matrix_id; }

    matrix matrix::operator+(const matrix& first)
    {
        std::vector<size_t> shape{m_matrix_ptr->get_shape().at(0),
                                  first.m_matrix_ptr->get_shape().at(1), 0};
        auto result_matrix_object_ptr = new matrix_object(shape);
        auto instruction =
                add(m_matrix_ptr, first.m_matrix_ptr, result_matrix_object_ptr);
        auto lock = std::unique_lock<std::mutex>(system_agent::m_fetch_schedule_mtx);
        system_agent::fetch_enable.wait(lock);
        return matrix(result_matrix_object_ptr);
    }

    matrix matrix::operator-(const matrix& first) { }

    matrix matrix::operator*(const matrix& first) { }

    bool matrix::operator==(const matrix& first) { }

    bool matrix::operator!=(const matrix& first) { return !(*this==first); }

    size_t matrix_hash_functor::operator()(const matrix& obj) const
    {
        return obj.get_id();
    }

    void matrix::wait_until_enabled() { }

} // namespace TakiMatrix