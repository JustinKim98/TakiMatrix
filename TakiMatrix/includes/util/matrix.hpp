//
// Created by jwkim98 on 19. 1. 6.
//

#ifndef TAKIMATRIX_MATRIX_HPP
#define TAKIMATRIX_MATRIX_HPP

#include <vector>

namespace TakiMatrix {
    class Matrix {
    public:
        explicit Matrix(std::vector<float>& rhs, const std::vector<int>& shape);

        Matrix(Matrix& matrix);

        Matrix& operator=(const Matrix& matrix);

        Matrix& operator+(const Matrix& matrix);

        Matrix& operator-(const Matrix& matrix);

        Matrix& operator*(const Matrix& matrix);

        Matrix& operator/(const Matrix& matrix);

        bool operator==(const Matrix& matrix) const;

        bool operator!=(const Matrix& matrix) const;

    private:
        void* getDataPtr();

        std::vector<float> data;

        std::vector<int> shape;
    };

    class Shape{
    private:
        std::vector<int> shape;
    };
} // namespace TakiMatrix

#endif // TAKIMATRIX_MATRIX_HPP
