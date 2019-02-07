//
// Created by jwkim98 on 19. 1. 14.
//

#include "../../../includes/cpu_emulator/system_agent/process.hpp"

namespace TakiMatrix::processor {

    process::~process()
    {
        auto thread_itr = m_thread_map.begin();
        for (; thread_itr!=m_thread_map.end(); ++thread_itr) {
            thread_itr->second.join();
        }
    }

    void process::reservation_table_insert(const instruction& instruction,
            std::thread::id tid)
    {
        auto rs_table_ptr = m_rs_map.find(tid);
        if (rs_table_ptr==m_rs_map.end())
            m_rs_map.insert(std::make_pair(tid, reservation_table()));
        auto thread_ptr = m_thread_map.find(tid);
        if (thread_ptr==m_thread_map.end())
            m_thread_map.insert(
                    std::make_pair(tid, std::thread([this, tid]() { schedule(tid); })));

        rs_table_ptr->second.insert(instruction);
    }

    std::deque<instruction> process::reservation_table_scan()
    {
        std::deque<instruction> start_list;
        m_reservation_table.scan(start_list);
        return start_list;
    }

    void process::schedule(std::thread::id tid)
    {
        while (m_is_activated) {
            std::deque<instruction> start_list;
            m_rs_map.at(tid).scan(start_list);
            auto itr = start_list.begin();

            for (; itr!=start_list.end(); ++itr) {
                auto inst_type = static_cast<size_t>(itr->type());
                m_execution_units.at(inst_type).at(0).allocate_instruction(*itr);
            }
        }
    }

    size_t process::smallest_queue_idx(instruction_type type){
        std::vector<execution_unit> units =
                m_execution_units.at(static_cast<size_t>(type));
        auto eu_ptr = units.begin();

        size_t minimum_buffer_size = -1;
        for (; eu_ptr!=units.end(); ++eu_ptr) {
            size_t buffer_size = eu_ptr->size();
            if (minimum_buffer_size>buffer_size)
                minimum_buffer_size = buffer_size;
        }
        return minimum_buffer_size;
    }


} // namespace TakiMatrix::processor