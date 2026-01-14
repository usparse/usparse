//
// Created by genshen to 2025-08-29.
//

#include <set>

#include "usparse_perf_record.h"

namespace usparse {
  namespace tuner {
    namespace internal {
      StatesPerfList tuner_state_perf_list;

      // config for output as stdout
      namespace console_config {
        int tune_case_title_length = 10;
        int tune_param_title_length = 12;
        bool counter_empty = true;
        int header_row_char_num = 0;
      } // namespace console_config

      std::vector<std::string> get_all_counters_name() {
        // get all user counters list
        std::set<std::string> user_counter_set;
        for (auto p : tuner_state_perf_list.state_time_lists) { // loop all case
          for (auto pp : p.second) {                            // loop all paramater
            for (auto &counter : pp.counters) {                 // loop all counters in a map.
              user_counter_set.insert(counter.first);
            }
          }
        }
        std::vector<std::string> user_counter_list(user_counter_set.begin(), user_counter_set.end());
        return std::move(user_counter_list);
      }

      void print_header_stdout() {
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

      void print_header_csv() {
        // print logo
        printf("\nuSparse Tuner summary:\n");
        // print header title.
        printf("%s,%s,%s,%s", "TuneCase", "TuneParams", "Time(μs)", "Iterations");

        std::vector<std::string> user_counter_list = get_all_counters_name();
        for (auto &counter_name : user_counter_list) {
          printf(",Counter:%s", counter_name.c_str());
        }
        printf("\n");
      }

      void print_header(const PERF_RECORD_OUTPUT output_target) {
        if (output_target == PERF_RECORD_OUTPUT_CSV) {
          print_header_csv();
        } else { // PERF_RECORD_OUTPUT_STDOUT
          print_header_stdout();
        }
      }

      void print_summary_stdout() {
        // loo all arguments:
        int i = 0;
        for (auto p : tuner_state_perf_list.state_time_lists) {
          // loop all tuning parameters:
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

      void print_summary_csv() {
        // get all counters name:
        std::vector<std::string> user_counter_list = get_all_counters_name();
        // loo all arguments:
        for (auto p : tuner_state_perf_list.state_time_lists) {
          // loop all tuning parameters:
          const std::string args_name = p.first;
          for (auto pp : p.second) {
            printf("%s,%s,%g,%d", args_name.c_str(), pp.tune_param_list_name.c_str(), pp.avarage_time_per_itl(),
                   pp.iterations);
            // loop all user counters. It keeps the same order as the csv header.
            for (const std::string &counter_name : user_counter_list) {
              std::map<std::string, usparse::tuner::Counter>::iterator iter = pp.counters.find(counter_name);
              if (iter != pp.counters.end()) {
                printf(",%s", iter->second.as_str(pp.iterations, pp.time_duration_us).c_str());
              } else {
                printf(","); // empty counter.
              }
            }
            printf("\n");
          }
        }
        printf("\n");
      }

      void print_summary(const PERF_RECORD_OUTPUT output_target) {
        if (output_target == PERF_RECORD_OUTPUT_CSV) {
          print_summary_csv();
        } else { // PERF_RECORD_OUTPUT_STDOUT
          print_summary_stdout();
        }
      }
    } // namespace internal
  } // namespace tuner
} // namespace usparse
