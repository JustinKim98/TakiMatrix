//
// Created by jwkim98 on 19. 1. 14.
//

#ifndef TAKIMATRIX_SYSTEM_AGENT_HPP
#define TAKIMATRIX_SYSTEM_AGENT_HPP

#include "../../util/matrix.hpp"
#include "../processor_util/instruction_set.hpp"
#include <deque>
#include <future>
#include <list>
#include <unordered_set>
#include <thread>
#include <vector>

namespace TakiMatrix::processor {

    enum class execution_units {
        add_1 = 0,
        add_2,
        mul_1,
        mul_2,
        transpose,
        dot_1,
        dot_2,
    };

    struct reorder_buffer_wrapper {
        reorder_buffer_wrapper(const isa& instruction, bool is_completed);

        isa m_instruction;
        bool m_is_completed = false;
    };

    const int num_execution_units = 7;

    class system_agent {
    public:
        /// inserts instruction to reservation table
        static bool rs_table_push(const isa& instruction);
        /**
         * @brief : scans m_rs_table and finds operations that are ready to be executed
         * those operations are deleted from m_rs_table
         * @param start_list
         */
        static void rs_table_scan(std::deque<isa>& start_list);
        /// pushes new instructions to reorder buffer
        static bool reorder_buffer_push(const isa& instruction);
        /**
         * @brief : commits pending instructions that has been completed, in order
         * applies changes to user code, and wakes up pending instruction in
         * reservation table
         */
        static void reorder_buffer_commit();

    private:
        /// max size of reservation table
        static const size_t max_rs_table_size;
        /// max size of reorder buffer
        static const size_t max_reorder_buffer_size;
        /// determines whether putting new instructions to rs table is enabled
        static std::condition_variable m_enable_fetch;
        /// determines whether taking out instructions from rs table is enabled
        static std::condition_variable m_enable_schedule;
        /// promise variables from scheduler to execution units
        static std::vector<std::promise<int>> m_scheduler_promises;
        /// used to commit changes to matrices used in user environment
        static std::unordered_set<matrix, matrix_hash_functor> m_matrix_map;
        /// reservation table for pending instructions
        static std::list<isa> m_rs_table;
        /// only one thread can access m_reorder_buffer(not protected by mutex)
        static std::deque<reorder_buffer_wrapper> m_reorder_buffer;

        static std::mutex m_rs_table_mtx;

        static std::mutex m_reorder_buffer_mtx;
    };


} // namespace TakiMatrix::processor

#endif // TAKIMATRIX_SYSTEM_AGENT_HPP