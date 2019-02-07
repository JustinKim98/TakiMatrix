/**
 * system_agent contains essential resources to perform ILP analysis
 *
 * system_agent includes reservation table, instruction queue, and execution
 * units defining this class will initialize a virtual processor
 *
 */

#ifndef TAKIMATRIX_PROCESS_HPP
#define TAKIMATRIX_PROCESS_HPP

#include "../execution_units/units/essential_unit.hpp"
#include "../front_end/instruction_queue.hpp"
#include "../front_end/reservation_table.hpp"
#include <deque>
#include <future>
#include <list>
#include <thread>
#include <unordered_map>
#include <unordered_set>
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

    const int num_execution_units = 7;

    class process {
    public:
        process(process&) = delete;

        ~process();

        /**
         *
         * @param instruction : instruction to insert
         * @param tid : thread_id of fetch thread
         */
        void reservation_table_insert(const instruction& instruction,
                std::thread::id tid);

        /**
         * scans reservation table and collects instructions without dependency
         * @return : instructions ready to be executed
         */
        std::deque<instruction> reservation_table_scan();

    private:

        void schedule(std::thread::id tid);

        size_t smallest_queue_idx(instruction_type type);

        std::vector<std::vector<execution_unit>> m_execution_units;
        /// true if this process is activated false otherwise
        std::atomic_bool m_is_activated;
        /// map of reservation_table
        std::unordered_map<std::thread::id, reservation_table> m_rs_map;
        /// reservation table to store pending instructions to be executed
        reservation_table m_reservation_table;
        /// thread object for running m_schedule;
        std::unordered_map<std::thread::id, std::thread> m_thread_map;
    };

} // namespace TakiMatrix::processor
#endif // TAKIMATRIX_PROCESS_HPP