//
// Created by jwkim98 on 19. 1. 13.
//

#include "../../../includes/cpu_emulator/front_end/instruction_cache.hpp"
#include <future>

namespace TakiMatrix::processor {
    void instruction_cache::push(const isa& instruction){
        std::lock_guard<std::mutex> guard(cache_mtx);
        m_cache_data.push_back(instruction);
    }

    isa instruction_cache::pop(){
        std::lock_guard<std::mutex> guard(cache_mtx);
        isa& instruction = m_cache_data.front();
        m_cache_data.pop_front();
        return instruction;
    }
}