/**
 * @file : matrix_object.cpp
 * @author : Justin Kim
 */

#include "../../../includes/cpu_emulator/processor_util/matrix_object.hpp"

namespace TakiMatrix::processor {

    matrix_object::matrix_object(const std::vector<size_t>& shape)
            :m_data(std::vector<float>(calculate_size(shape), 0)), m_shape(shape),
             m_size(m_data.size()), m_matrix_object_id(++matrix_object_count) { }

    matrix_object::matrix_object(const std::vector<float>& data,
            const std::vector<size_t>& shape)
            :m_data(data), m_shape(shape), m_size(m_data.size()*sizeof(float)),
             m_matrix_object_id(++matrix_object_count) { }

    matrix_object::matrix_object(const matrix_object& rhs)
            :m_data(rhs.m_data), m_shape(rhs.m_shape), m_size(rhs.m_size),
             m_matrix_object_id(++matrix_object_count) { }

    matrix_object::~matrix_object()
    {
        if (device_ptr!=nullptr)
            cudaFree(device_ptr);
    }

    bool matrix_object::operator==(const matrix_object& first) const
    {
        return first.m_data==m_data;
    }

    size_t matrix_object::get_id() const { return m_matrix_object_id; }

    void matrix_object::set_ready() { m_is_completed = true; }

    bool matrix_object::is_ready() { return m_is_completed; }

    std::vector<size_t> matrix_object::get_shape() { return m_shape; }

    size_t matrix_object::get_size() { return m_size; }

    float* matrix_object::get_data_ptr() { return m_data.data(); }

} // namespace TakiMatrix::processor