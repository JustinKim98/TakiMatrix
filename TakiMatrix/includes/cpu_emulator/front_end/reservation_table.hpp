//
// Created by jwkim98 on 19/01/17.
//

#ifndef TAKIMATRIX_RESERVATION_TABLE_HPP
#define TAKIMATRIX_RESERVATION_TABLE_HPP

#include "../../cpu_emulator/processor_util/matrix_object.hpp"
#include "../processor_util/instruction_set.hpp"
#include <condition_variable>
#include <deque>
#include <list>
#include <thread>
#include <unordered_set>

namespace TakiMatrix::processor {
    class reservation_table {
    public:
        /**
         * implements reservation table that can be concurrently accessed
         * @param table_size : maximum size of the table
         */
        explicit reservation_table(size_t table_size = 100);
        /**
         * inserts instruction to the end of the reservation table
         * @param instruction : instruction to insert
         */
        void insert(const instruction& instruction);
        /**
         * scans the reservation table, and puts executable instructions to
         * start_list (collects instructions without read-after-write dependency)
         * @param start_list : reference of deque to insert executable instructions
         */
        void scan(std::deque<instruction>& start_list);

    private:
        const size_t m_maximum_table_size;
        std::list<instruction> m_reservation_table;
        std::deque<instruction> m_dependency_free_instruction_table;
        std::mutex m_mtx;
        std::condition_variable m_cond;
    };
} // namespace TakiMatrix::processor
#endif // TAKIMATRIX_RESERVATION_TABLE_HPP