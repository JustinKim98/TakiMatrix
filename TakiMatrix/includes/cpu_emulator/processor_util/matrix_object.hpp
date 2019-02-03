/**
 * @file : matrix_object.hpp
 * @author : Justin Kim
 *
 * matrix_object stores shape and data of the matrix
 */

#ifndef TAKIMATRIX_MATRIX_OBJECT_HPP
#define TAKIMATRIX_MATRIX_OBJECT_HPP

#include "../system_agent/utility.hpp"
#include <atomic>
#include <vector>

namespace TakiMatrix::processor {
    std::atomic<size_t> matrix_object_count;

    class matrix_object {
    public:
        /**
         * constructs matrix_object with empty data(initialized to 0) with size of shape
         *
         * @param shape : shape of new empty matrix_object
         */
        explicit matrix_object(const std::vector<size_t>& shape);
        /**
         * constructs matrix_object with vector
         * (Not associated with matrix object in user code)
         *
         * @param data : data for this matrix
         * @param shape : shape of the data
         */
        matrix_object(const std::vector<float>& data,
                const std::vector<size_t>& shape);
        /**
         * copy constructor for matrix_object
         * assigns new m_matrix_object_id
         *
         * @param rhs : matrix_object to copy from
         */
        matrix_object(const matrix_object& rhs);
        /**
         * equality operator
         * compares m_data of this and new matrix
         *
         * @param first
         * @return : true if equal false otherwise
         */
        bool operator==(const matrix_object& first) const;
        /**
         * gets id of this matrix_object
         * @return : m_matrix_id of this matrix_object
         */
        size_t get_id() const;
        /**
         * sets m_is_completed to true(false as default)
         */
        void set_ready();
        /**
         * returns if this matrix is ready
         * @return : m_is_completed of this matrix_object
         */
        bool is_ready();
        /**
         * returns shape of this matrix as 3 dimensional vector
         * @return : m_shape of this matrix
         */
        std::vector<size_t> get_shape();

        size_t get_size();

        float* get_data_ptr();

    private:
        /// data of this matrix
        std::vector<float> m_data;
        /// shape of this matrix
        std::vector<size_t> m_shape;
        /// data size in bytes
        size_t m_size = 0;
        /// unique id of this matrix_object
        size_t m_matrix_object_id = 0;
        /// set true if instruction is completed, and ready to be committed
        bool m_is_completed = false;
    };
} // namespace TakiMatrix::processor

#endif // TAKIMATRIX_MATRIX_HPP
