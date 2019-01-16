//
// Created by jwkim98 on 19/01/16.
//

#include "../../../includes/cpu_emulator/system_agent/utility.hpp"

namespace TakiMatrix::processor{
    size_t calculate_size(const std::vector<size_t>& shape)
    {
        size_t size = 1;
        for (size_t elem : shape) {
            size *= elem;
        }
        return size;
    };
}