//
// Created by jwkim98 on 19/01/14.
//

#ifndef TAKIMATRIX_MATRIX_HPP
#define TAKIMATRIX_MATRIX_HPP

#include "../cpu_emulator/processor_util/matrix_object.hpp"
#include <memory>

namespace TakiMatrix {

    size_t matrix_id = 0;

    class matrix {
    public:
        matrix(const std::vector<float>& data, const std::vector<size_t>& shape);

        matrix(matrix& new_matrix);

        matrix(matrix&& new_matrix) noexcept;

        ~matrix();

        size_t get_id() const;

    private:
        std::unique_ptr<processor::matrix_object> m_matrix_ptr;

        size_t m_matrix_id;
    };

    struct compare_matrix {
        bool operator()(const matrix& first, const matrix& second)
        {
            return first.get_id()>second.get_id();
        }
    };

    struct matrix_hash_functor {
        size_t operator()(const matrix& obj) const;
    };
} // namespace TakiMatrix

#endif // TAKIMATRIX_MATRIX_HPP
