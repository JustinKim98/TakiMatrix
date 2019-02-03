/**
 * system_agent contains essential resources to perform ILP analysis
 *
 * system_agent includes reservation table, instruction queue, and execution
 * units defining this class will initialize a virtual processor
 *
 */

#ifndef TAKIMATRIX_PROCESS_HPP
#define TAKIMATRIX_PROCESS_HPP

#include "../front_end/instruction_queue.hpp"
#include "../front_end/reservation_table.hpp"
#include <deque>
#include <future>
#include <list>
#include <thread>
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
        /**
         * pushes instruction to concurrent instruction queue for this process
         * @param instruction : instruction to push into the queue
         */
        void instruction_queue_push(const instruction& instruction);
        /**
         * pops instruction from the concurrent instruction queue for this process
         * @return : instruction to be executed
         */
        instruction instruction_queue_pop();
        /**
         * waits until instruction queue is empty
         * this method can be used for synchronization or branches
         */
        void instruction_queue_wait_until_empty();
        /**
         * inserts instruction to the reservation table
         * @param instruction : instruction to insert
         */
        instruction_queue& get_instruction_queue();

        void reservation_table_insert(const instruction& instruction);
        /**
         * scans reservation table and collects instructions without dependency
         * @return : instructions ready to be executed
         */
        std::deque<instruction> reservation_table_scan();
    private:
        /// instruction queue to store fetched instructions
        instruction_queue m_instruction_queue;
        /// reservation table to store pending instructions to be executed
        reservation_table m_reservation_table;
        /// thread object for running m_fetch
        std::thread m_fetch_thread;
        /// thread object for running m_schedule;
        std::thread m_schedule_thread;
        /// gets instruction from m_instruction_queue and puts it into reservation table
        std::function<void(void)> m_fetch =  [this](){
            instruction inst = m_instruction_queue.pop();
            m_reservation_table.insert(inst);
        };
        /// allocates dependency-free instructions to execution units
        std::function<void(void)> m_schedule = [this](){
            std::deque<instruction> start_list;
            m_reservation_table.scan(start_list);
            //TODO: put instructions into corresponding execution units
        };

    };

} // namespace TakiMatrix::processor
#endif // TAKIMATRIX_PROCESS_HPP