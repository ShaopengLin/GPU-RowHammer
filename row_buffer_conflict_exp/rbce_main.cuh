#include "rbce_n_conflict_exp.cuh"
#include <stdint.h>
#include <string>
#ifndef GPU_ROWHAMMER_RBCE_MAIN_H
#define GPU_ROWHAMMER_RBCE_MAIN_H

namespace rbce
{
const static std::string TWO_CONFLICT = "2c";
const static std::string FIND_ROW = "fr";
const static std::string FIND_ROW_FILE = "frf";
const static std::string FIND_SAME_BANK_ADDR = "sb";
const static std::string BANK_SET_FILE_NAME = "SAME_BANK_ADDR.txt";
void run_two_conflict_exp(N_Conflict &nc_test);
void find_n_rows(N_Conflict &nc_test);
} // namespace rbce

#endif /* GPU_ROWHAMMER_RBCE_MAIN_H */