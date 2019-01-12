//
// Created by jwkim98 on 19. 1. 12.
//

#include "../../includes/util/matrix.hpp"

namespace TakiMatrix {
    matrix::matrix(const std::vector<float>& data, const std::vector<size_t>& shape)
            :data(data), shape(shape) { }

    matrix::matrix(const matrix& rhs)
    {
        this->data = rhs.data;
        this->shape = rhs.shape;
    }


} // namespace TakiMatrix