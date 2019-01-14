//
// Created by jwkim98 on 19. 1. 14.
//

#include "../../../includes/cpu_emulator/system_agent/system_agent.hpp"
#include "../../../includes/cpu_emulator/processor_util/instruction_set.hpp"

namespace TakiMatrix::processor {
    reorder_buffer_wrapper::reorder_buffer_wrapper(const isa& instruction,
            bool is_completed)
            :m_instruction(instruction), m_is_completed(is_completed) { }

    void system_agent::instruction_cache_push(const isa& instruction)
    {
        std::lock_guard<std::mutex> guard(m_instruction_queue_mtx);
        m_instruction_queue.push_back(instruction);
    }

    isa system_agent::instruction_cache_pop()
    {
        std::lock_guard<std::mutex> guard(m_instruction_queue_mtx);
        isa& instruction = m_instruction_queue.front();
        m_instruction_queue.pop_front();
        return instruction;
    }

    void system_agent::reorder_buffer_push(const isa& instruction)
    {
        reorder_buffer_wrapper temp(instruction, false);

        m_reorder_buffer.emplace_back(temp);
    }

    void reorder_buffer_commit() { }
} // namespace TakiMatrix::processor