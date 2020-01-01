//
// Created by jwkim98 on 19. 1. 13.
//

#ifndef TAKIMATRIX_INSTRUCTION_QUEUE_HPP
#define TAKIMATRIX_INSTRUCTION_QUEUE_HPP

#include "../processor_util/instruction_set.hpp"
#include <condition_variable>
#include <thread>
#include <deque>


namespace TakiMatrix::processor {
    class instruction_queue {
    public:
        /**
         * this class implements instruction cache
         * that can be concurrently accessed by producer-consumer method
         * @param queue_size : maximum queue size that this class can hold
         */
        explicit instruction_queue(size_t queue_size = 100);
        /**
         * waits until m_instruction_queue is empty
         * this method can be used to wait until specific instruction is executed
         */
        void wait_until_empty();
        /**
         * waits until operation on matrix_ptr is completed
         * @param matrix_ptr : ptr to matrix_object to wait for
         */
        void wait_for(std::shared_ptr<matrix_object> matrix_ptr);
        /**
         * adds instruction into the front of m_instruction_queue
         * @param instruction : instruction to add
         */
        void push(const instruction& instruction);
        /**
         * gets last instruction in the m_instruction_queue and removes it from the m_instruction_queue
         * @return : last instruction in the m_instruction_queue
         */
        instruction pop();

        size_t size();

    private:
        /// m_maximum_queue_size
        const size_t m_maximum_queue_size;
        /// queue storing instructions
        std::deque<instruction> m_instruction_queue;

        std::mutex instruction_queue_mtx;

        std::condition_variable m_cond;
    };
} // namespace TakiMatrix::processor

#endif // TAKIMATRIX_INSTRUCTION_QUEUE_HPP
