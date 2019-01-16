//
// Created by jwkim98 on 19. 1. 14.
//

#include "../../../includes/cpu_emulator/system_agent/system_agent.hpp"
#include "../../../includes/cpu_emulator/processor_util/instruction_set.hpp"

namespace TakiMatrix::processor {

    const size_t system_agent::max_rs_table_size = 150;

    const size_t system_agent::max_reorder_buffer_size = 300;
/// determines whether putting new instructions to rs table is enabled
    std::condition_variable system_agent::m_enable_fetch;
/// determines whether taking out instructions from rs table is enabled
    std::condition_variable system_agent::m_enable_schedule;
/// promise variables from scheduler to execution units
    std::vector<std::promise<int>> system_agent::m_scheduler_promises;
/// reservation table for pending instructions
    std::list<isa> system_agent::m_rs_table;
/// only one thread can access m_reorder_buffer(not protected by mutex)
    std::deque<reorder_buffer_wrapper> system_agent::m_reorder_buffer;

    std::mutex system_agent::m_fetch_schedule_mtx;

    std::mutex system_agent::m_reorder_buffer_mtx;

    reorder_buffer_wrapper::reorder_buffer_wrapper(const isa& instruction,
            bool is_completed)
            :m_instruction(instruction), m_is_completed(is_completed) { }

    bool system_agent::rs_table_push(
            const TakiMatrix::processor::isa& instruction)
    {
        if (m_rs_table.size()>max_rs_table_size)
            return false;
        m_rs_table.emplace_back(instruction);
        return true;
    }

    void system_agent::rs_table_scan(std::deque<isa>& start_list)
    {
        std::unordered_set<size_t> write_operated_matrix_object_ids;

        auto has_previous_write_operation =
                [&write_operated_matrix_object_ids](size_t matrix_object_id) {
                    if (write_operated_matrix_object_ids.find(matrix_object_id)==
                            write_operated_matrix_object_ids.end())
                        return false;
                    return true;
                };

        auto rs_table_itr = m_rs_table.begin();
        for (; rs_table_itr!=m_rs_table.end(); rs_table_itr++) {
            size_t matrix_object_id = rs_table_itr->get_result_ptr()->get_id();
            if (!has_previous_write_operation(matrix_object_id)) {
                start_list.emplace_back(*rs_table_itr);
                rs_table_itr = m_rs_table.erase(rs_table_itr);
                rs_table_itr--;
            }
        }
    }

    bool system_agent::reorder_buffer_push(const isa& instruction)
    {
        reorder_buffer_wrapper temp(instruction, false);
        std::lock_guard<std::mutex> guard(m_reorder_buffer_mtx);
        if (m_reorder_buffer.size()>max_reorder_buffer_size)
            return false;
        m_reorder_buffer.emplace_back(temp);
        return true;
    }

    void system_agent::reorder_buffer_commit() { }
} // namespace TakiMatrix::processor