//
// Created by genshen to 2025-08-29.
//

#ifndef USPARSE_TUNER_USPARSE_TUNER_UTILS_HPP
#define USPARSE_TUNER_USPARSE_TUNER_UTILS_HPP

#include <sstream>
#include <string>
#include <tuple>

namespace usparse {
  namespace tuner {
    namespace utils {
      template <typename... Args> std::string tupleToString(const std::tuple<Args...> &t) {
        if constexpr (sizeof...(Args) == 0) {
          return "";
        } else {
          std::ostringstream stream;
          std::apply(
              [&stream](const auto &...elements) {
                size_t n = 0;
                ((stream << elements << (++n != sizeof...(Args) ? "/" : "")), ...);
              },
              t);

          return "/" + stream.str(); // format: TuneName/args
        }
      }

    } // namespace utils
  } // namespace tuner
} // namespace usparse

#endif // USPARSE_TUNER_USPARSE_TUNER_UTILS_HPP
