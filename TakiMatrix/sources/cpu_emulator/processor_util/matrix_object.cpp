//
// Created by jwkim98 on 19. 1. 12.
//

#include "../../../includes/cpu_emulator/processor_util/matrix_object.hpp"
#include <cassert>

namespace TakiMatrix::processor {
    matrix_object::matrix_object(const std::vector<float>& data,
            const std::vector<size_t>& shape)
            :m_data(data), m_shape(shape)
    {
        size_t size = 0;
        assert(shape.size()==3);

        for (auto elem : shape) {
            size *= elem;
        }
        assert(size==data.size());
        data_size = data.size()*sizeof(float);
        m_matrix_object_id = matrix_object_id++;
    }

    matrix_object::matrix_object(const std::vector<float>& data,
            const std::vector<size_t>& shape,
            size_t origin_id)
    {
        size_t size = 0;
        assert(shape.size()==3);

        for (auto elem : shape) {
            size *= elem;
        }
        assert(size==data.size());
        data_size = data.size()*sizeof(float);
        m_matrix_object_id = matrix_object_id++;

        m_has_origin = true;
        m_origin_id = origin_id;
    }

    matrix_object::matrix_object(const matrix_object& rhs)
    {
        this->m_data = rhs.m_data;
        this->m_shape = rhs.m_shape;
        data_size = rhs.m_data.size()*sizeof(float);
    }

    size_t matrix_object::get_id() const { return m_matrix_object_id; }

    size_t matrix_object::get_origin_id() const { return m_origin_id; }

} // namespace TakiMatrix::processor