#pragma once
#include "types.hpp"

__global__ void division_by_rowA(const csrValueType alpha,const size_t dev_A_row,const size_t dev_B_col,const csrIndexType* dptr_offset_A,
const csrIndexType* dptr_colindex_A,const csrValueType* dptr_value_A,const csrIndexType* dptr_offset_B,
const csrIndexType* dptr_colindex_B,const csrValueType* dptr_value_B,csrValueType* tmp_dev_c);