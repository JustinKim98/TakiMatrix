//
// Created by jwkim98 on 19. 1. 12.
//

#include "../../includes/util/matrix.hpp"
#include <cassert>

namespace TakiMatrix {
    matrix::matrix(const std::vector<float>& data, const std::vector<size_t>& shape)
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

    matrix::matrix(const matrix& rhs)
    {
        this->data = rhs.data;
        this->shape = rhs.shape;
        data_size = rhs.data.size()*sizeof(float);
    }

} // namespace TakiMatrix