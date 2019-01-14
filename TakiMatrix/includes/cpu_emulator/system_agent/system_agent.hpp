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
        isa instruction;
        bool is_coimpleted = false;
    };

    const int num_execution_units = 7;

    class system_agent {
    public:
        void instruction_cache_push(const isa& instruction);

        isa instruction_cache_pop();

        void reorder_buffer_push(const isa& instruction);

        isa reorder_buffer_commit();
    private:
        ///determines whether putting new instructions to rs table is enabled
        static std::condition_variable m_enable_fetch;
        ///determines whether taking out instructions from rs table is enabled
        static std::condition_variable m_enable_schedule;
        ///
        static std::vector<std::promise<int>> m_scheduler_promises;
        ///used to commit changes to matrices used in user environment
        static std::map<size_t, matrix> m_matrix_map;

        static std::promise<int> m_fetch_to_schedule;

        static std::list<isa>  m_rs_table;

        static std::deque<isa> m_reorder_buffer;

        static std::deque<isa> m_register_renaming_table;

        static std::deque<isa> m_cache_data;

        static std::mutex m_cache_mtx;

        static std::mutex m_rs_table_mtx;
    };
}


#endif // TAKIMATRIX_SYSTEM_AGENT_HPP