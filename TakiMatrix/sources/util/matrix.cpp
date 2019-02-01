//
// Created by jwkim98 on 19/01/14.
//

#include "../../includes/util/matrix.hpp"

namespace TakiMatrix {
    matrix::matrix(std::reference_wrapper<process> processor, const std::vector<float>& data,
            const std::vector<size_t>& shape)
            :m_processor(processor)
    {
        m_matrix_ptr = std::make_shared<matrix_object>(data, shape);
    }

    matrix::matrix(std::reference_wrapper<process> processor,
            std::shared_ptr<matrix_object>& matrix_object_ptr)
            :m_processor(processor)
    {
        m_matrix_ptr = matrix_object_ptr;
    }

    matrix matrix::operator+(const matrix& first)
    {
        std::vector<size_t> shape = m_matrix_ptr->get_shape();

        std::shared_ptr<matrix_object> result_ptr =
                std::make_shared<matrix_object>(shape);
        auto temp =
                instruction(instruction_type::add, m_matrix_ptr, first.m_matrix_ptr, result_ptr);
        m_processor.get().instruction_queue_push(temp);
        return matrix(m_processor, result_ptr);
    }

    matrix matrix::operator-(const matrix& first)
    {
        std::vector<size_t> shape = m_matrix_ptr->get_shape();
        std::shared_ptr<matrix_object> result_ptr =
                std::make_shared<matrix_object>(shape);
        auto temp =
                instruction(instruction_type::sub, m_matrix_ptr, first.m_matrix_ptr, result_ptr);
        m_processor.get().instruction_queue_push(temp);
        return matrix(m_processor, result_ptr);
    }

    matrix matrix::operator*(const matrix& first)
    {
        std::vector<size_t> shape{m_matrix_ptr->get_shape().at(0),
                                  first.m_matrix_ptr->get_shape().at(1), 0};
        std::shared_ptr<matrix_object> result_ptr =
                std::make_shared<matrix_object>(shape);
        auto temp =
                instruction(instruction_type::mul, m_matrix_ptr, first.m_matrix_ptr, result_ptr);
        m_processor.get().instruction_queue_push(temp);
        return matrix(m_processor, result_ptr);
    }

    bool matrix::operator==(const matrix& first)
    {
        m_processor.get().get_instruction_queue().wait_for(first.matrix_ptr());
        return *(first.m_matrix_ptr)==*m_matrix_ptr;
    }

    bool matrix::operator!=(const matrix& first) { return !(*this==first); }

    std::shared_ptr<matrix_object> matrix::matrix_ptr() const{
        if(!m_matrix_ptr->is_ready())
            m_processor.get().get_instruction_queue().wait_for(m_matrix_ptr);
        return m_matrix_ptr;
    }

} // namespace TakiMatrix