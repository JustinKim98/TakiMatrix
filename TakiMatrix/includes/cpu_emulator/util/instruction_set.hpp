//
// Created by jwkim98 on 19. 1. 12.
//

#ifndef TAKIMATRIX_INSTRUCTION_SET_HPP
#define TAKIMATRIX_INSTRUCTION_SET_HPP

namespace cpu_emulator{
    enum class isa_type{
        add,
        sub,
        mul,
        malloc_gpu,
        malloc_cpu,
        copy_d2h,
        copy_h2d,
        free_cpu,
        free_gpu,
        transpose,
    };
    
    class isa{
    public:

    };
}

#endif //TAKIMATRIX_INSTRUCTION_SET_HPP
