//
// Created by jwkim98 on 19. 1. 12.
//

#ifndef TAKIMATRIX_MATRIX_HPP
#define TAKIMATRIX_MATRIX_HPP

#include <cstdio>
#include <vector>

namespace TakiMatrix {
    class matrix {
    public:
        matrix() = default;

        matrix(const std::vector<float>& data, const std::vector<size_t>& shape);

        matrix(const matrix& rhs);

    private:
        std::vector<float> data;
        std::vector<size_t> shape;
    };
} // namespace TakiMatrix

#endif // TAKIMATRIX_MATRIX_HPP
