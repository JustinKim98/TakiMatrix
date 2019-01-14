//
// Created by jwkim98 on 19. 1. 14.
//

#ifndef TAKIMATRIX_SYSTEM_AGENT_HPP
#define TAKIMATRIX_SYSTEM_AGENT_HPP


#include <future>
#include <thread>
#include <vector>
#include <deque>
#include <list>
#include <map>
#include "../processor_util/instruction_set.hpp"
#include "../../util/matrix.hpp"

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

    struct reorder_buffer_wrapper{
        reorder_buffer_wrapper(const isa& instruction, bool is_completed);
        isa m_instruction;
        bool m_is_completed = false;
    };

    const int num_execution_units = 7;

    class system_agent {
    public:
        void instruction_cache_push(const isa& instruction);

        isa instruction_cache_pop();

        void reorder_buffer_push(const isa& instruction);

        void reorder_buffer_commit();
    private:
        ///determines whether putting new instructions to rs table is enabled
        static std::condition_variable m_enable_fetch;
        ///determines whether taking out instructions from rs table is enabled
        static std::condition_variable m_enable_schedule;
        ///promise variables from scheduler to execution units
        static std::vector<std::promise<int>> m_scheduler_promises;
        ///used to commit changes to matrices used in user environment
        static std::map<size_t, matrix> m_matrix_map;
        ///reservation table for pending instructions
        static std::list<isa>  m_rs_table;
        ///only one thread can access m_reorder_buffer(not protected by mutex)
        static std::deque<reorder_buffer_wrapper> m_reorder_buffer;

        static std::deque<isa> m_instruction_queue;

        static std::mutex m_instruction_queue_mtx;

        static std::mutex m_rs_table_mtx;
    };
}


#endif // TAKIMATRIX_SYSTEM_AGENT_HPP