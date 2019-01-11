//
// Created by jwkim98 on 19. 1. 6.
//

#include "../../includes/util/matrix.hpp"

namespace TakiMatrix {

    Matrix::Matrix(std::vector<float>& rhs, const std::vector<size_t>& shape)
    {
        int shape_size = 1;
        for (int elem : shape) {
            shape_size *= elem;
        }

        if (rhs.size()==shape_size)
            data = rhs;
        else {
            // TODO print and log error
        }
        this->shape = shape;
    }

    Matrix::Matrix(Matrix& matrix) { data = matrix.data; }

    Matrix& Matrix::operator=(const Matrix& matrix)
    {
        data = matrix.data;
        return *this;
    }

    Matrix& Matrix::operator+(const Matrix& matrix) { }

    Matrix& Matrix::operator-(const Matrix& matrix) { }

    Matrix& Matrix::operator*(const Matrix& matrix) { }

    Matrix& Matrix::operator/(const Matrix& matrix) { }

    bool Matrix::operator==(const Matrix& matrix) const
    {
        return data==matrix.data;
    }

    bool Matrix::operator!=(const Matrix& matrix) const
    {
        return !(matrix==*this);
    }

    void Matrix::assignData(float* first, size_t size, const std::vector<size_t>& shape)
    {

        data = std::vector<float>(size);
        data.assign(first, first+size);
        this->shape = shape;
    }
} // namespace TakiMatrix