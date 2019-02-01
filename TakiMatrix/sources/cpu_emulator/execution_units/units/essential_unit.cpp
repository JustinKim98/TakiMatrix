/**
 * @file : essential_unit.cpp
 */

#include "../../../../includes/cpu_emulator/execution_units/units/essential_unit.hpp"

namespace TakiMatrix::processor {

    execution_unit::~execution_unit() { m_thread.join(); }

    void execution_unit::allocate_instruction(instruction& inst)
    {
        m_instruction_buffer.push(inst);
    }

    void execution_unit::start()
    {
        m_thread = std::thread([this] { process(); });
    }

    void add_eu::process()
    {
        //TODO: think about conditions about loop
        while (m_enabled) {
            call_add(m_instruction_buffer.pop());
        }
    }

    void add_eu::call_add(instruction&& inst)
    {
        float* device_operand_a;
        float* device_operand_b; // stores operand_second and result
        size_t operand_a_size = inst.first_operand_ptr()->get_data_size();
        size_t operand_b_size = inst.second_operand_ptr()->get_data_size();

        cudaMalloc((void**) &device_operand_a, operand_a_size);
        cudaMalloc((void**) &device_operand_b, operand_b_size);

        cudaMemcpy(device_operand_a, inst.first_operand_ptr().get(), operand_a_size,
                cudaMemcpyHostToDevice);
        cudaMemcpy(device_operand_b, inst.second_operand_ptr().get(), operand_a_size,
                cudaMemcpyHostToDevice);
        // TODO: call the kernel here
        cudaMemcpy(inst.result_ptr().get(), device_operand_b, operand_b_size,
                cudaMemcpyDeviceToHost);
        cudaFree(device_operand_a);
        cudaFree(device_operand_b);
    }

    void sub_eu::process()
    {
        while (m_enabled) {
            call_sub(m_instruction_buffer.pop());
        }
    }

    void sub_eu::call_sub(instruction&& inst)
    {
        float* device_operand_a;
        float* device_operand_b; // stores operand_second and result
        size_t operand_a_size = inst.first_operand_ptr()->get_data_size();
        size_t operand_b_size = inst.second_operand_ptr()->get_data_size();

        cudaMalloc((void**) &device_operand_a, operand_a_size);
        cudaMalloc((void**) &device_operand_b, operand_b_size);

        cudaMemcpy(device_operand_a, inst.first_operand_ptr().get(), operand_a_size,
                cudaMemcpyHostToDevice);
        cudaMemcpy(device_operand_b, inst.second_operand_ptr().get(), operand_a_size,
                cudaMemcpyHostToDevice);
        // TODO: call the kernel here
        cudaMemcpy(inst.result_ptr().get(), device_operand_b, operand_b_size,
                cudaMemcpyDeviceToHost);
        cudaFree(device_operand_a);
        cudaFree(device_operand_b);
    }

    void mul_eu::process()
    {
        while (m_enabled) {
            call_mul(m_instruction_buffer.pop());
        }
    }

    void mul_eu::call_mul(instruction&& inst)
    {
        float* device_operand_a;
        float* device_operand_b;
        float* device_result;
        size_t operand_a_size = inst.first_operand_ptr()->get_data_size();
        size_t operand_b_size = inst.second_operand_ptr()->get_data_size();
        size_t result_size = inst.result_ptr()->get_data_size();

        cudaMalloc((void**) &device_operand_a, operand_a_size);
        cudaMalloc((void**) &device_operand_b, operand_b_size);
        cudaMalloc((void**) &device_result, result_size);

        cudaMemcpy(device_operand_a, inst.first_operand_ptr().get(), operand_a_size,
                cudaMemcpyHostToDevice);
        cudaMemcpy(device_operand_b, inst.second_operand_ptr().get(), operand_a_size,
                cudaMemcpyHostToDevice);
        // TODO: call the kernel here
        cudaMemcpy(inst.result_ptr().get(), device_result, result_size,
                cudaMemcpyDeviceToHost);
        cudaFree(device_operand_a);
        cudaFree(device_operand_b);
        cudaFree(device_result);
    }

} // namespace TakiMatrix::processor