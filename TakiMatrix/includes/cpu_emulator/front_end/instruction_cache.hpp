//
// Created by jwkim98 on 19. 1. 13.
//

#ifndef TAKIMATRIX_INSTRUCTION_CACHE_HPP
#define TAKIMATRIX_INSTRUCTION_CACHE_HPP
#include "../processor_util/instruction_set.hpp"
#include <deque>
#include <mutex>

namespace TakiMatrix::processor{
    class instruction_cache{
        static void push(const isa& instruction);

        static isa pop();
    private:
        static std::deque<isa> m_cache_data;
        static std::mutex cache_mtx;
};

    std::deque<isa> instruction_cache::m_cache_data = std::deque<isa>();
    std::mutex instruction_cache::cache_mtx;
}
#endif //TAKIMATRIX_INSTRUCTION_CACHE_HPP