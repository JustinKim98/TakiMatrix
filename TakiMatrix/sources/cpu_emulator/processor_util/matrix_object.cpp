//
// Created by jwkim98 on 19. 1. 12.
//

#include "../../../includes/cpu_emulator/processor_util/matrix_object.hpp"
#include <cassert>

namespace TakiMatrix::processor{
    matrix_object::matrix_object(const std::vector<float>& data, const std::vector<size_t>& shape)
            :data(data), shape(shape)
    {
        size_t size = 0;
        assert(shape.size()==3);

        for (auto elem : shape) {
            size *= elem;
        }
        assert(size==data.size());
        data_size = data.size()*sizeof(float);
    }

    matrix_object::matrix_object(const matrix_object& rhs)
    {
        this->data = rhs.data;
        this->shape = rhs.shape;
        data_size = rhs.data.size()*sizeof(float);
    }

} // namespace TakiMatrix