#ifndef SPGEMM_HIP_TYPES_H
#define SPGEMM_HIP_TYPES_H

enum hip_sparse_operation { hip_operation_none = 0, hip_operation_transpose = 1 };

typedef float sp_data_type;     // type of single precision float point number.
typedef double dp_data_type;    // type of double precision float point number.
typedef dp_data_type data_type; // default data type for float point number.
typedef int idx_type;           // type of index.

#endif //SPGEMM_HIP_TYPES_H