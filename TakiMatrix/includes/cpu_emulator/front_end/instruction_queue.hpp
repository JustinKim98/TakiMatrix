//
// Created by jwkim98 on 19. 1. 13.
//

#ifndef TAKIMATRIX_INSTRUCTION_QUEUE_HPP
#define TAKIMATRIX_INSTRUCTION_QUEUE_HPP

#include "../../cpu_emulator/processor_util/matrix_object.hpp"
#include "../processor_util/instruction_set.hpp"
#include <condition_variable>
#include <thread>
#include <deque>


namespace TakiMatrix::processor {
    class instruction_queue {
    public:
        /**
         * @brief : this class implements instruction cache
         * that can be concurrently accessed by producer-consumer method
         * @param queue_size : maximum queue size that this class can hold
         */
        explicit instruction_queue(size_t queue_size = 100);
        void wait_until_empty();
        /**
         * @brief : pushes instructions into the queue
         * @param instruction : instruction to execute
         */
        void push(const instruction& instruction);
        /**
         * @brief : pops instructions from the queue
         * @return : popped instruction
         */
        instruction pop();

    private:

        const size_t m_maximum_queue_size;

        std::deque<instruction> m_instruction_queue;

        std::mutex instruction_queue_mtx;

        std::condition_variable m_cond;
    };
} // namespace TakiMatrix::processor

#endif // TAKIMATRIX_INSTRUCTION_QUEUE_HPP
