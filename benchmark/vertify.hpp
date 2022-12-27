//
// Created by luxu on 2022/12/12.
//

#ifndef SPGEMM_BENCHMARK_VERTIFY_HPP
#define SPGEMM_BENCHMARK_VERTIFY_HPP

template <typename I, typename T> 
void vertify(csr_mtx<I, T> &matrix, csr_mtx<I, T> &compare_matrix) {
  bool bEqual = true;

  for (size_t row = 0; row < compare_matrix.rows; row++) {
    size_t off1_l = matrix.row_ptr[row];
    size_t off1_r = matrix.row_ptr[row + 1];
    size_t off2_l = compare_matrix.row_ptr[row];
    size_t off2_r = compare_matrix.row_ptr[row + 1];

    size_t num1 = 0;
    size_t num2 = 0;
    size_t num1_max = off1_r - off1_l;
    size_t num2_max = off2_r - off2_l;
    size_t num_max = max(num1_max, num2_max);

    while ((num1 < num_max) && (num2 < num_max)) {
      size_t col1 = matrix.col_index[off1_l + num1];
      size_t col2 = compare_matrix.col_index[off2_l + num2];
      if (num1 < num1_max && num2 < num2_max) {
        if (col1 == col2) {
          double fab_diff = fabs(matrix.values[off1_l + num1] - compare_matrix.values[off2_l + num2]);
          double fab_mat = fabs(compare_matrix.values[off2_l + num2]);

          if ((fab_mat > 1.0) && (fab_diff >= fab_mat * 1e-7)) {
            bEqual = false;
          } else if ((fab_mat <= 1.0) && (fab_diff >= 1e-7)) {
            bEqual = false;
          }

          if (!bEqual) {
#ifdef DEVICE_SIDE_VERIFY_FLAG
            std::cout << "[" << row << "," << col1 << "] : "
                      << "fab_diff " << fab_diff << " ustb_spgemm_value " << matrix.values[off1_l + num1]
                      << " rocsparse_value " << compare_matrix.values[off2_l + num2] << std::endl;
#else
            std::cout << "Host vertify is not supported!" << std::endl;
            exit(0);
#endif
            std::cout << "off " << off1_l << " " << off1_r << " " << off2_l << " " << off2_r << std::endl;
            std::cout << "col " << col1 << " " << col2 << " index " << num1 << " " << num2 << std::endl;
            std::cout << "Failed verification, please check your code!" << std::endl;
            return;
          }
          num1++;
          num2++;
        } else if (col1 < col2) {
          double fab_diff = fabs(matrix.values[off1_l + num1]);

          if (fab_diff >= 1e-7) {
            bEqual = false;
          }

          if (!bEqual) {
#ifdef DEVICE_SIDE_VERIFY_FLAG
            std::cout << "[" << row << "," << col1 << "] : "
                      << "fab_diff " << fab_diff << " ustb_spgemm_value " << matrix.values[off1_l + num1]
                      << " rocsparse_value " << compare_matrix.values[off2_l + num2] << std::endl;
#else
            std::cout << "Host vertify is not supported!" << std::endl;
            exit(0);
#endif

            std::cout << "off " << off1_l << " " << off1_r << " " << off2_l << " " << off2_r << std::endl;
            std::cout << "col " << col1 << " " << col2 << " index " << num1 << " " << num2 << std::endl;
            std::cout << "Failed verification, please check your code!" << std::endl;
            return;
          }
          num1++;
        } else if (col1 > col2) {
          double fab_diff = fabs(compare_matrix.values[off2_l + num2]);

          if (fab_diff >= 1e-7) {
            bEqual = false;
          }

          if (!bEqual) {
#ifdef DEVICE_SIDE_VERIFY_FLAG
            std::cout << "[" << row << "," << col1 << "] : "
                      << "fab_diff " << fab_diff << " ustb_spgemm_value " << matrix.values[off1_l + num1]
                      << " rocsparse_value " << compare_matrix.values[off2_l + num2] << std::endl;
#else
            std::cout << "Host vertify is not supported!" << std::endl;
            exit(0);
#endif
            std::cout << "off " << off1_l << " " << off1_r << " " << off2_l << " " << off2_r << std::endl;
            std::cout << "col " << col1 << " " << col2 << " index " << num1 << " " << num2 << std::endl;
            std::cout << "Failed verification, please check your code!" << std::endl;
            return;
          }
          num2++;
        }
      } else if (num1 >= num1_max) {
        double fab_diff = fabs(compare_matrix.values[off2_l + num2]);

        if (fab_diff >= 1e-7) {
          bEqual = false;
        }

        if (!bEqual) {
#ifdef DEVICE_SIDE_VERIFY_FLAG
          std::cout << "[" << row << "," << col1 << "] : "
                    << "fab_diff " << fab_diff << " ustb_spgemm_value no_value  rocsparse_value "
                    << compare_matrix.values[off2_l + num2] << std::endl;
#else
          std::cout << "Host vertify is not supported!" << std::endl;
          exit(0);
#endif
          std::cout << "off " << off1_l << " " << off1_r << " " << off2_l << " " << off2_r << std::endl;
          std::cout << "col no_device_col " << col2 << " index " << num1 << " " << num2 << std::endl;
          std::cout << "Failed verification, please check your code!" << std::endl;
          return;
        }
        num2++;
      } else if (num2 >= num2_max) {
        double fab_diff = fabs(matrix.values[off1_l + num1]);

        if (fab_diff >= 1e-7) {
          bEqual = false;
        }

        if (!bEqual) {
#ifdef DEVICE_SIDE_VERIFY_FLAG
          std::cout << "[" << row << "," << col1 << "] : "
                    << "fab_diff " << fab_diff << " ustb_spgemm_value " << matrix.values[off1_l + num1]
                    << " rocsparse_value no_value" << std::endl;
#else
          std::cout << "Host vertify is not supported!" << std::endl;
          exit(0);
#endif
          std::cout << "off " << off1_l << " " << off1_r << " " << off2_l << " " << off2_r << std::endl;
          std::cout << "col " << col1 << " no_compare_col"
                    << " index " << num1 << " " << num2 << std::endl;
          std::cout << "Failed verification, please check your code!" << std::endl;
          return;
        }
        num1++;
      }
    }
  }

  std::cout << "Congratulations, verification pass! " << std::endl;
}

#endif // SPGEMM_BENCHMARK_VERTIFY_HPP
