//
// Created by jwkim98 on 19. 1. 13.
//

#ifndef TAKIMATRIX_INSTRUCTION_FETCH_HPP
#define TAKIMATRIX_INSTRUCTION_FETCH_HPP
#include <deque>
#include <future>
#include <thread>
#include <condition_variable>
#include "../processor_util/instruction_set.hpp"
#include "../front_end/instruction_cache.hpp"
#include "../../cpu_emulator/processor_util/matrix_object.hpp"

namespace TakiMatrix::processor{


    void fetch(std::promise<int>* to_set){

    }

}

#endif //TAKIMATRIX_INSTRUCTION_FETCH_HPP
