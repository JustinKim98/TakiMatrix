//
// Created by jwkim98 on 19. 1. 6.
//

#ifndef TAKIMATRIX_MATRIX_HPP
#define TAKIMATRIX_MATRIX_HPP

#include <cstddef>
#include <vector>

namespace TakiMatrix {
    class Matrix {
    public:
        explicit Matrix(std::vector<float>& rhs, const std::vector<size_t>& shape);

        Matrix(Matrix& matrix);

        Matrix& operator=(const Matrix& matrix);

        Matrix& operator+(const Matrix& matrix);

        Matrix& operator-(const Matrix& matrix);

        Matrix& operator*(const Matrix& matrix);

        Matrix& operator/(const Matrix& matrix);

        bool operator==(const Matrix& matrix) const;

        bool operator!=(const Matrix& matrix) const;

        void assignData(float* first, size_t size, const std::vector<size_t>& shape);

        void* getDataPtr();

        const std::vector<float>& getData() const;

        const std::vector<size_t>& getShape() const;

        size_t getDataSize() const;

    private:
        bool readEnabled = false;

        bool writeEnabled = false;

        std::vector<float> data;

        std::vector<size_t> shape;

        size_t dataSize;
    };

    class Shape {
    private:
        std::vector<int> shape;
    };
} // namespace TakiMatrix

#endif // TAKIMATRIX_MATRIX_HPP
