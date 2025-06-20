
#include <fstream>
#include <iostream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

typedef char *BytePtr;

template <typename I, typename T> struct CSR {
  I rows; // matrix's row
  I cols; // matrix's col
  I nnz;  // The number of non-zero elements in a matrix
  I *row_ptr;
  I *col_index;
  T *values;
};

template <typename I, typename T> CSR<I, T> load_mat(const std::string &mtx_path) {
  std::ifstream fin(mtx_path, std::ios::in | std::ios::binary);
  if (!fin.good()) {
    std::cerr << "file open failed, file: " << mtx_path << std::endl;
    CSR<I, T> csr_data_c;
    return csr_data_c;
  }
  // read header: magic number
  int32_t magic_num;
  fin.read((BytePtr)(&(magic_num)), sizeof(magic_num));
  if (magic_num != 0x20211015) {
    std::cerr << "read file failed with mismatch magic number, file:" << mtx_path << std::endl;
    CSR<I, T> csr_data_c;
    return csr_data_c;
  }
  // read header: binary format
  int32_t bin_format;
  fin.read((BytePtr)(&(bin_format)), sizeof(bin_format));
  if (bin_format != 0x2) {
    std::cerr << "we only support bin file version 2, bin file:" << mtx_path << std::endl;
    CSR<I, T> csr_data_c;
    return csr_data_c;
  }
  // read header: value type
  typedef int32_t tp_val_tpye;
  tp_val_tpye val_type;
  fin.read((BytePtr)(&(val_type)), sizeof(tp_val_tpye));
  constexpr tp_val_tpye TP_BOOL = 1;
  constexpr tp_val_tpye TP_INT = 2;
  constexpr tp_val_tpye TP_FLOAT = 3;
  constexpr tp_val_tpye TP_COMPLEX = 4;
  if (val_type != TP_BOOL && val_type != TP_INT && val_type != TP_FLOAT && val_type != TP_COMPLEX) {
    std::cerr << "matrix value type not supported, bin file:" << mtx_path << std::endl;
    CSR<I, T> csr_data_c;
    return csr_data_c;
  }

  CSR<I, T> csr_data;
  // read csr matrix
  //    I &_r = P::rows;
  //    I &_c = P::cols;
  //    I &_nnz = P::nnz;
  fin.read((BytePtr)(&(csr_data.rows)), sizeof(I)); // todo: make sure storage type is 4 bytes.
  fin.read((BytePtr)(&(csr_data.cols)), sizeof(I));
  fin.read((BytePtr)(&(csr_data.nnz)), sizeof(I));

  csr_data.row_ptr = new I[csr_data.rows + 1];
  csr_data.col_index = new I[csr_data.nnz];
  csr_data.values = new T[csr_data.nnz];

  fin.read((BytePtr)(csr_data.row_ptr), sizeof(I) * (csr_data.rows + 1));
  fin.read((BytePtr)(csr_data.col_index), sizeof(I) * csr_data.nnz);

  // read csr matrix: value part
  if (val_type == TP_BOOL) {
    for (int k = 0; k < csr_data.nnz; k++) {
      csr_data.values[k] = 1.0;
    }
  } else if (val_type == TP_INT) {
    int32_t *temp_int = new int32_t[csr_data.nnz];
    fin.read((BytePtr)(temp_int), sizeof(T) * csr_data.nnz);
    for (int k = 0; k < csr_data.nnz; k++) {
      csr_data.values[k] = temp_int[k];
    }
    delete[] temp_int;
  } else { // float or complex, read directly.
    fin.read((BytePtr)(csr_data.values), sizeof(T) * csr_data.nnz);
  }
  fin.close();
  return csr_data;
}
