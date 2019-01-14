//
// Created by jwkim98 on 19. 1. 12.
//

#ifndef TAKIMATRIX_MATRIX_OBJECT_HPP
#define TAKIMATRIX_MATRIX_OBJECT_HPP

#include <cstdio>
#include <vector>

namespace TakiMatrix::processor{
    class matrix_object {
    public:
        matrix_object() = default;

        matrix_object(const std::vector<float>& data, const std::vector<size_t>& shape);

        matrix_object(const matrix_object& rhs);

    private:
        std::vector<float> data;
        std::vector<size_t> shape;
        ///data size in bytes
        size_t data_size = 0;
    };
} // namespace TakiMatrix

#endif // TAKIMATRIX_MATRIX_HPP
