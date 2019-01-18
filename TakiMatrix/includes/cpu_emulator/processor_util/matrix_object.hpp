//
// Created by jwkim98 on 19. 1. 12.
//

#ifndef TAKIMATRIX_MATRIX_OBJECT_HPP
#define TAKIMATRIX_MATRIX_OBJECT_HPP

#include "../system_agent/utility.hpp"
#include <cassert>
#include <cstdio>
#include <vector>

namespace TakiMatrix::processor {
    size_t matrix_object_id = 0;

    class matrix_object {
    public:
        matrix_object() = default;

        matrix_object(const std::vector<size_t>& shape);

        /**
         * @brief : constructor for matrix_objects
         * (Not associated with matrix object in user code)
         * @param data : data for this matrix
         * @param shape : shape of the data
         * @param has_origin : true if this matrix_object is non-temporary false
         * otherwise
         */
        matrix_object(const std::vector<float>& data,
                const std::vector<size_t>& shape, bool has_origin = false);

        matrix_object(const matrix_object& rhs);

        bool operator==(const matrix_object& first) const;

        size_t get_id() const;

        size_t get_origin_id() const;

        void set_ready();

        bool is_ready();

        bool has_origin();

        std::vector<size_t> get_shape();

    private:
        std::vector<float> m_data;
        std::vector<size_t> m_shape;
        /// data size in bytes
        size_t data_size = 0;
        /// unique id of this matrix_object
        size_t m_matrix_object_id;
        /// true if this object is non-temporary
        bool m_has_origin = false;
        /// id of origin matrix (if it has one)
        size_t m_origin_id = 0;
        /// set true if instruction is completed, and ready to be committed
        bool m_is_completed = false;
    };
} // namespace TakiMatrix::processor

#endif // TAKIMATRIX_MATRIX_HPP
