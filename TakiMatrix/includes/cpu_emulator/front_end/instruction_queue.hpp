//
// Created by jwkim98 on 19. 1. 13.
//

#ifndef TAKIMATRIX_INSTRUCTION_FETCH_HPP
#define TAKIMATRIX_INSTRUCTION_FETCH_HPP

#include "../../cpu_emulator/processor_util/matrix_object.hpp"
#include "../processor_util/instruction_set.hpp"
#include <condition_variable>
#include <deque>
#include <future>
#include <thread>

namespace TakiMatrix::processor {
    class instruction_queue {
    public:
        /**
         * @brief : this class implements instruction cache
         * that can be concurrently accessed by producer-consumer method
         * @param queue_size : maximum queue size that this class can hold
         */
        explicit instruction_queue(size_t queue_size = 1000);
        /**
         * @brief : pushes instructions into the queue
         * @param instruction : instruction to execute
         */
        void push(isa& instruction);
        /**
         * @brief : pops instructions from the queue
         * @return : popped instruction
         */
        isa pop();

    private:
        const size_t m_default_queue_size;
        std::deque<isa> m_instruction_queue;
        std::mutex instruction_queue_mtx;
        std::condition_variable m_cond;
    };
} // namespace TakiMatrix::processor

#endif // TAKIMATRIX_INSTRUCTION_FETCH_HPP
