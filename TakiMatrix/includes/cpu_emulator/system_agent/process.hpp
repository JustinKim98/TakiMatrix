/**
 * system_agent contains essential resources to perform ILP analysis
 *
 * system_agent includes reservation table, instruction queue, and execution units
 * defining this class will initialize a virtual processor
 *
 */

#ifndef TAKIMATRIX_PROCESS_HPP
#define TAKIMATRIX_PROCESS_HPP

#include "../processor_util/instruction_set.hpp"
#include "../front_end/instruction_queue.hpp"
#include "../front_end/reservation_table.hpp"
#include <deque>
#include <future>
#include <list>
#include <unordered_set>
#include <thread>
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
        process();

        void execute(std::thread to_execute);

        void execute_async(std::thread to_execute);

        void synchronize();

        void instruction_queue_push(const isa& instruction);

        isa instruction_queue_pop();

        void instruction_queue_wait_until_empty();

        void reservation_table_insert(const isa& instruction);

        std::deque<isa> reservation_table_scan();

    private:

        /// instruction queue to store fetched instructions
        instruction_queue m_instruction_queue;
        /// reservation table to store pending instructions to be executed
        reservation_table m_reservation_table;
    };

} // namespace TakiMatrix::processor
#endif // TAKIMATRIX_PROCESS_HPP