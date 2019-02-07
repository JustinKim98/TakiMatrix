//
// Created by jwkim98 on 19. 1. 14.
//

#include "../../../includes/cpu_emulator/system_agent/compute_unit.hpp"

namespace TakiMatrix::processor {
    compute_unit::compute_unit()
    {
        //TODO: make move constructors && operators for execution units!
   //     std::vector<execution_unit> add_units(10,add_eu());
   //     std::vector<execution_unit> sub_units(10, sub_eu());
   //     std::vector<execution_unit> mul_units(10, mul_eu());
   //     std::vector<execution_unit> dot_units(10, dot_eu());
   //     std::vector<std::vector<execution_unit>> execution_unit_vectors{
   //             add_units, sub_units, mul_units, dot_units};
        //m_execution_unit_vectors = execution_unit_vectors;
    }

    compute_unit::~compute_unit()
    {
        auto thread_itr = m_thread_map.begin();
        for (; thread_itr!=m_thread_map.end(); ++thread_itr) {
            thread_itr->second.join();
        }
    }

    void compute_unit::reservation_table_insert(const instruction& instruction,
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

    std::deque<instruction> compute_unit::reservation_table_scan()
    {
        std::deque<instruction> start_list;
        m_reservation_table.scan(start_list);
        return start_list;
    }


    void compute_unit::schedule(std::thread::id tid)
    {
        while (m_is_activated) {
            std::deque<instruction> start_list;
            m_rs_map.at(tid).scan(start_list);
            auto itr = start_list.begin();

            for (; itr!=start_list.end(); ++itr) {
                auto instruction_id = static_cast<size_t>(itr->type());
                size_t unit_id = smallest_unit_idx(itr->type());
                m_execution_unit_vectors.at(instruction_id)
                        .at(unit_id)
                        .allocate_instruction(*itr);
            }
        }
    }

    size_t compute_unit::smallest_unit_idx(instruction_type type)
    {
        std::vector<execution_unit>& units =
                m_execution_unit_vectors.at(static_cast<size_t>(type));
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