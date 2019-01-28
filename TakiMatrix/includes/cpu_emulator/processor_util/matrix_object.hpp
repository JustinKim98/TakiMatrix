//
// Created by jwkim98 on 19. 1. 12.
//

#ifndef TAKIMATRIX_MATRIX_OBJECT_HPP
#define TAKIMATRIX_MATRIX_OBJECT_HPP

#include "../system_agent/utility.hpp"
#include <atomic>
#include <cassert>
#include <cstdio>
#include <vector>

namespace TakiMatrix::processor {
    std::atomic<size_t> matrix_object_count;

    class matrix_object {
    public:
        matrix_object() = default;
        /**
         * constructs matrix_object with empty data(initialized to 0) with size of shape
         * @param shape : shape of new empty matrix_object
         */
        explicit matrix_object(const std::vector<size_t>& shape);
        /**
         * constructs matrix_object with vector
         * (Not associated with matrix object in user code)
         * @param data : data for this matrix
         * @param shape : shape of the data
         */
        matrix_object(const std::vector<float>& data,
                const std::vector<size_t>& shape);

        matrix_object(const matrix_object& rhs);

        bool operator==(const matrix_object& first) const;

        size_t get_id() const;

        void set_ready();

        bool is_ready();

        std::vector<size_t> get_shape();

    private:
        std::vector<float> m_data;
        std::vector<size_t> m_shape;
        /// data size in bytes
        size_t m_data_size = 0;
        /// unique id of this matrix_object
        size_t m_matrix_object_id = 0;
        /// set true if instruction is completed, and ready to be committed
        bool m_is_completed = false;
    };
} // namespace TakiMatrix::processor

#endif // TAKIMATRIX_MATRIX_HPP
