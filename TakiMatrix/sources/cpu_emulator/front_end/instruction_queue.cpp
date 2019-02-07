//
// Created by jwkim98 on 19. 1. 13.
//

#include "../../../includes/cpu_emulator/front_end/instruction_queue.hpp"

namespace TakiMatrix::processor {
    instruction_queue::instruction_queue(size_t queue_size)
            :m_maximum_queue_size(queue_size) { }

    void instruction_queue::push(const instruction& instruction)
    {
        std::unique_lock<std::mutex> lock(instruction_queue_mtx);
        m_cond.wait(lock, [this]() {
            return m_instruction_queue.size()<m_maximum_queue_size;
        });
        m_instruction_queue.emplace_back(instruction);
        lock.unlock();
        m_cond.notify_all();
    }

    instruction instruction_queue::pop()
    {
        std::unique_lock<std::mutex> lock(instruction_queue_mtx);
        m_cond.wait(lock, [this]() { return !m_instruction_queue.empty(); });
        instruction instruction = m_instruction_queue.front();
        m_instruction_queue.pop_front();
        lock.unlock();
        m_cond.notify_all();
    }

    void instruction_queue::wait_until_empty()
    {
        std::unique_lock<std::mutex> lock(instruction_queue_mtx);
        m_cond.wait(lock, [this]() { return m_instruction_queue.empty(); });
    }

    void instruction_queue::wait_for(std::shared_ptr<matrix_object> matrix_ptr)
    {
        std::unique_lock<std::mutex> lock(instruction_queue_mtx);
        m_cond.wait(lock, [&matrix_ptr]() { return matrix_ptr->is_ready(); });
    }

    size_t instruction_queue::size() { return m_instruction_queue.size(); }

} // namespace TakiMatrix::processor