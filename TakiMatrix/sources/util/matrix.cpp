//
// Created by jwkim98 on 19/01/14.
//

#include "../../includes/util/matrix.hpp"

namespace TakiMatrix {
    matrix::matrix(process& processor, const std::vector<float>& data,
            const std::vector<size_t>& shape)
            :m_processor(processor)
    {
        m_matrix_id = matrix_id++;
        m_matrix_ptr = new matrix_object(data, shape, true);
    }

    matrix::matrix(process& processor, matrix_object* matrix_object_ptr)
            :m_processor(processor)
    {
        m_matrix_ptr = matrix_object_ptr;
        m_matrix_id = matrix_id++;
    }

    matrix::matrix(matrix& new_matrix)
            :m_processor(new_matrix.m_processor)
    {
        processor::matrix_object temp = *new_matrix.m_matrix_ptr;
        m_matrix_ptr = new matrix_object(temp);
        m_matrix_id = matrix_id++;
    }

    matrix::matrix(matrix&& new_matrix) noexcept
            :m_processor(new_matrix.m_processor)
    {
        m_matrix_ptr = new_matrix.m_matrix_ptr;
        m_matrix_id = matrix_id++;
    }

    matrix::~matrix()
    {
        if (m_matrix_ptr->has_origin())
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
        m_processor.get().instruction_queue_push(instruction);
        return matrix(m_processor, result_matrix_object_ptr);
    }

    matrix matrix::operator-(const matrix& first)
    {
        std::vector<size_t> shape{m_matrix_ptr->get_shape().at(0),
                                  first.m_matrix_ptr->get_shape().at(1), 0};
        auto result_matrix_object_ptr = new matrix_object(shape);
        auto instruction =
                sub(m_matrix_ptr, first.m_matrix_ptr, result_matrix_object_ptr);
        m_processor.get().instruction_queue_push(instruction);
        return matrix(m_processor, result_matrix_object_ptr);
    }

    matrix matrix::operator*(const matrix& first)
    {
        std::vector<size_t> shape{m_matrix_ptr->get_shape().at(0),
                                  first.m_matrix_ptr->get_shape().at(1), 0};
        auto result_matrix_object_ptr = new matrix_object(shape);
        auto instruction =
                mul(m_matrix_ptr, first.m_matrix_ptr, result_matrix_object_ptr);
        m_processor.get().instruction_queue_push(instruction);
        return matrix(m_processor, result_matrix_object_ptr);
    }

    bool matrix::operator==(const matrix& first)
    {
        m_processor.get().instruction_queue_wait_until_empty();
        return *(first.m_matrix_ptr)==*m_matrix_ptr;
    }

    bool matrix::operator!=(const matrix& first) { return !(*this==first); }

    size_t matrix_hash_functor::operator()(const matrix& obj) const
    {
        return obj.get_id();
    }

} // namespace TakiMatrix