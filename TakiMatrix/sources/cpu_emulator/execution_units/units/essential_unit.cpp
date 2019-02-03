/**
 * @file : essential_unit.cpp
 */

#include "../../../../includes/cpu_emulator/execution_units/units/essential_unit.hpp"

execution_unit::~execution_unit() { m_thread.join(); }

void execution_unit::allocate_instruction(instruction& inst)
{
    m_instruction_buffer.push(inst);
}

void execution_unit::start()
{
    m_thread = std::thread([this] { process(); });
}

void execution_unit::process()
{
    while (m_enabled) {
        caller(m_instruction_buffer.pop());
    }
}

std::function<void(instruction&&)> add_eu::add = [](instruction&& inst) {
    size_t operand_a_size = inst.first_operand_ptr()->get_size();
    size_t operand_b_size = inst.second_operand_ptr()->get_size();
    size_t size = 0;

    if (operand_a_size==operand_b_size) {
        size = operand_a_size;
    }
    // TODO: exception handling for size mismatch

    float* operand_a_ptr = inst.first_operand_ptr()->get_data_ptr();
    float* operand_b_ptr = inst.second_operand_ptr()->get_data_ptr();

    kernel::call_add(operand_a_ptr, operand_b_ptr, size);
};

std::function<void(instruction&&)> sub_eu::sub = [](instruction&& inst) {
    size_t operand_a_size = inst.first_operand_ptr()->get_size();
    size_t operand_b_size = inst.second_operand_ptr()->get_size();
    size_t size = 0;

    if (operand_a_size==operand_b_size) {
        size = operand_a_size;
    }
    // TODO: exception handling for size mismatch

    float* operand_a_ptr = inst.first_operand_ptr()->get_data_ptr();
    float* operand_b_ptr = inst.second_operand_ptr()->get_data_ptr();

    kernel::call_sub(operand_a_ptr, operand_b_ptr, size);
};

std::function<void(instruction&&)> mul_eu::mul = [](instruction&& inst) {
    size_t operand_a_size = inst.first_operand_ptr()->get_size();
    size_t operand_b_size = inst.second_operand_ptr()->get_size();
    size_t result_size = inst.result_ptr()->get_size();

    float* operand_a_ptr = inst.first_operand_ptr()->get_data_ptr();
    float* operand_b_ptr = inst.second_operand_ptr()->get_data_ptr();
    float* result_ptr = inst.result_ptr()->get_data_ptr();

    std::vector<size_t> operand_a_shape = inst.first_operand_ptr()->get_shape();
    std::vector<size_t> operand_b_shape = inst.second_operand_ptr()->get_shape();

    size_t middle_size = 0;
    size_t dimension_size = 0;
    if (operand_a_shape.at(1)==operand_a_shape.at(0))
        middle_size = operand_a_shape.at(1);

    if(operand_a_shape.at(2) == operand_b_shape.at(2))
        dimension_size = operand_a_shape.at(2);

    kernel::call_mul(operand_a_ptr, operand_b_ptr, result_ptr, middle_size,
            operand_a_shape.at(0), operand_b_shape.at(1), dimension_size);
};

std::function<void(instruction&&)> dot_eu::dot = [](instruction&& inst) {
    size_t operand_a_size = inst.first_operand_ptr()->get_size();
    size_t size = operand_a_size;

    std::function<float(float)> functor = inst.functor();

    // TODO: exception handling for size mismatch

    float* operand_a_ptr = inst.first_operand_ptr()->get_data_ptr();

    kernel::call_dot(operand_a_ptr, functor, size);
};
