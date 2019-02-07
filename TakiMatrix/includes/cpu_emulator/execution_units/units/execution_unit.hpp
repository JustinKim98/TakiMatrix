/**
 *
 * @file : essential_unit.hpp
 *
 * execution units for running instructions
 *
 * each instruction has concurrent instruction queue and thread for sending them
 * to gpu kernel
 */

#ifndef TAKIMATRIX_ADD_HPP
#define TAKIMATRIX_ADD_HPP

#include "../../front_end/instruction_queue.hpp"
#include "../kernels/kernal_caller.h"
#include <deque>
#include <thread>

using namespace TakiMatrix::processor;

/**
 * base class for essential execution units
 */
class execution_unit {
public:
    virtual ~execution_unit();

    /**
     * inserts instruction into m_instruction_queue
     * @param inst : instruction to allocate
     */
    virtual void allocate_instruction(instruction& inst);

    /**
     * starts thread for executing instructions in m_instruction_queue
     */
    void start();

    /**
     * enables executing thread
     */
    void enable() { m_enabled = true; }

    /**
     * disables executing thread
     */
    void disable() { m_enabled = false; }

    /**
     * returns instruction buffer size
     * @return : size of internal instruction buffer
     */
    size_t size() { return m_instruction_buffer.size()}

protected:
    /// disable construction of default constructor
    execution_unit() = default;
    /// process
    virtual void process();
    /// concurrent instruction queue to store temporary instructions
    instruction_queue m_instruction_buffer;
    /// thread object for executing instructions
    std::thread m_thread;
    /// stops m_thread if false
    bool m_enabled = true;
    /// function for calling the gpu kernel
    std::function<void(instruction&&)> caller;
};

class add_eu : public execution_unit {
public:
    add_eu() { caller = add; }

private:
    static std::function<void(instruction&&)> add;
};

class sub_eu : public execution_unit {
public:
    sub_eu() { caller = sub; }

private:
    static std::function<void(instruction&&)> sub;
};

class mul_eu : public execution_unit {
public:
    mul_eu() { caller = mul; }

private:
    static std::function<void(instruction&&)> mul;
};

class dot_eu : public execution_unit {
public:
    dot_eu() { caller = dot; }

private:
    static std::function<void(instruction&&)> dot;
};

#endif // TAKIMATRIX_ADD_HPP
