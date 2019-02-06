//
// Created by jwkim98 on 19. 1. 14.
//

#include "../../../includes/cpu_emulator/system_agent/process.hpp"

namespace TakiMatrix::processor {

    void process::instruction_queue_push(
            const instruction& instruction)
    {
        m_instruction_queue.push(instruction);
    }

    instruction process::instruction_queue_pop() { return m_instruction_queue.pop(); }

    void process::reservation_table_insert(const instruction& instruction)
    {
        m_reservation_table.insert(instruction);
    }

    void process::instruction_queue_wait_until_empty()
    {
        m_instruction_queue.wait_until_empty();
    }

    instruction_queue& process::get_instruction_queue(){
        return m_instruction_queue;
    }

    std::deque<instruction> process::reservation_table_scan()
    {
        std::deque<instruction> start_list;
        m_reservation_table.scan(start_list);
        return start_list;
    }

} // namespace TakiMatrix::processor