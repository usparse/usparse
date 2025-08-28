//
// Created by genshen to 2025-08-30.
//

#include "usparse_tuner.h"
#include <cstddef>

namespace usparse {
  namespace tuner {
    namespace internal {
      std::vector<std::function<void()>> *tuner_entrypointers = nullptr;

      void append_tuner(const std::string &tuner_name, std::function<void()> func) {
        if (tuner_entrypointers == nullptr) {
          tuner_entrypointers = new std::vector<std::function<void()>>();
        }
        tuner_entrypointers->emplace_back(func);
        if (TUNE_DEBUG) {
          printf("add tuner: %s.\n", tuner_name.c_str());
        }
      }

      private_tuner *register_tuner(const std::string &tuner_name, std::function<void()> func) {
        append_tuner(tuner_name, func);
        auto unused = new private_tuner{};
        return unused;
      }

      void run_all_tuners() {
        for (auto func : (*tuner_entrypointers)) {
          func();
        }
      }
    } // namespace internal
  } // namespace tuner
} // namespace usparse
