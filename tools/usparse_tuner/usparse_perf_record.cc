//
// Created by genshen to 2025-08-29.
//

#include "usparse_perf_record.h"

namespace usparse {
  namespace tuner {
    namespace internal {
      StatesPerfList tuner_state_perf_list;

      namespace console_config {
        int tune_case_title_length = 10;
        int tune_param_title_length = 12;
        bool counter_empty = true;
        int header_row_char_num = 0;
      } // namespace console_config

      void print_header() {
        // find the max length of case name string (1st colmn in the summary table)
        // and tune param string (2nd colmn in the summary table).
        for (auto p : tuner_state_perf_list.state_time_lists) {
          const int cur_case_str_length = p.first.size();
          if (cur_case_str_length > console_config::tune_case_title_length) {
            console_config::tune_case_title_length = cur_case_str_length;
          }
          for (auto pp : p.second) {
            const int cur_tune_param_title_str_length = pp.tune_param_list_name.size();
            if (cur_tune_param_title_str_length > console_config::tune_param_title_length) {
              console_config::tune_param_title_length = cur_tune_param_title_str_length;
            }
            if (!(pp.counters.empty())) {
              console_config::counter_empty = false;
            }
          }
        }

        console_config::header_row_char_num =
            console_config::tune_case_title_length + console_config::tune_param_title_length + 15 + 12 + 3; // 3 space
        if (!(console_config::counter_empty)) {
          console_config::header_row_char_num += (18 + 1);
        }

        // print logo
        printf("\nuSparse Tuner summary:\n");
        // print header line.
        for (int i = 0; i < console_config::header_row_char_num; i++) {
          printf("-");
        }
        printf("\n");
        // print header title.
        printf("%-*s %-*s %15s %12s", static_cast<int>(console_config::tune_case_title_length), "TuneCase",
               console_config::tune_param_title_length, "TuneParams", "Time(μs)", "Iterations");

        // for user counters
        if (!(console_config::counter_empty)) {
          printf(" %13s\n", "Counters");
        } else {
          printf("\n");
        }
        // print header line (end).
        for (int i = 0; i < console_config::header_row_char_num; i++) {
          printf("-");
        }
        printf("\n");
      }

      void print_summary() {
        // loo all arguments:
        int i = 0;
        for (auto p : tuner_state_perf_list.state_time_lists) {
          // loop all input args:
          const std::string args_name = p.first;
          for (auto pp : p.second) {
            printf("%-*s %-*s %15g %12d", static_cast<int>(console_config::tune_case_title_length), args_name.c_str(),
                   console_config::tune_param_title_length, pp.tune_param_list_name.c_str(), pp.avarage_time_per_itl(),
                   pp.iterations);
            if (!(pp.counters.empty())) {
              for (auto counter : pp.counters) {
                printf(" %s: %s", counter.first.c_str(),
                       counter.second.as_str(pp.iterations, pp.time_duration_us).c_str());
              }
            }
            printf("\n");
          }

          i++;
          if (i != tuner_state_perf_list.state_time_lists.size()) {
            printf("----------\n"); // print split line of each args.
          }
        }
        // head line end
        for (int i = 0; i < console_config::header_row_char_num; i++) {
          printf("-");
        }
        printf("\n");
      }
    } // namespace internal
  } // namespace tuner
} // namespace usparse
