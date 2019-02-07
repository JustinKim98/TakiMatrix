//
// Created by jwkim98 on 19/01/17.
//

#include "../../../includes/cpu_emulator/front_end/reservation_table.hpp"

namespace TakiMatrix::processor {

    reservation_table::reservation_table(size_t table_size)
            :m_maximum_table_size(table_size) { }

    void reservation_table::insert(const instruction& instruction)
    {
        std::unique_lock<std::mutex> lock(m_mtx);
        m_cond.wait(lock, [this]() {
            return m_reservation_table.size()<m_maximum_table_size;
        });
        m_reservation_table.emplace_back(instruction);
        lock.unlock();
        m_cond.notify_all();
    }

    reservation_table::reservation_table(reservation_table&& rs_table) noexcept
            :m_maximum_table_size(rs_table.m_maximum_table_size) { }

    void reservation_table::scan(std::deque<instruction>& start_list)
    {
        /**
         * to avoid RAW dependency problem, it must check matrix_object whether it has
         * write operation previously
         */
        std::unordered_set<size_t> write_operated_matrix_object_ids;

        auto has_previous_write_operation =
                [&write_operated_matrix_object_ids](size_t matrix_object_id) {
                    if (write_operated_matrix_object_ids.find(matrix_object_id)==
                            write_operated_matrix_object_ids.end())
                        return false;
                    return true;
                };

        std::unique_lock<std::mutex> lock(m_mtx);
        m_cond.wait(lock, [this]() { return !m_reservation_table.empty(); });

        auto rs_table_itr = m_reservation_table.begin();
        for (; rs_table_itr!=m_reservation_table.end(); rs_table_itr++) {
            size_t first_id = rs_table_itr->first_operand_ptr()->get_id();
            size_t second_id = rs_table_itr->second_operand_ptr()->get_id();
            size_t result_id = rs_table_itr->result_ptr()->get_id();
            write_operated_matrix_object_ids.emplace(result_id);

            if (!has_previous_write_operation(first_id) &&
                    !has_previous_write_operation(second_id)) {
                start_list.emplace_back(*rs_table_itr);
                rs_table_itr = m_reservation_table.erase(rs_table_itr);
                rs_table_itr--;
            }
        }
        lock.unlock();
        m_cond.notify_all();
    }

} // namespace TakiMatrix::processor