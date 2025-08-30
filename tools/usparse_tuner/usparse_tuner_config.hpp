//
// Created by genshen to 2025-08-29.
//

#ifndef USPARSE_TUNER_USPARSE_TUNER_CONFIG_HPP
#define USPARSE_TUNER_USPARSE_TUNER_CONFIG_HPP

#include <iostream>
#include <tuple>
#include <utility>

namespace usparse {
  namespace tuner {

    template <typename... TypeArgs> struct Config {

      using TypeArgTuple = std::tuple<TypeArgs...>;
      using TypeArgsList = std::vector<TypeArgTuple>;

    private:
      TypeArgsList arg_list;
      std::string tune_name = "default";

      int args_num = 0;
      int iterations = 128; // default iterations.
    public:
      static Config *NewDefaultConfig() {
        auto c = new Config("default");
        return c;
      }

      Config() {}

      Config(const std::string &_tune_name) : tune_name(_tune_name) {}

      void setTuneName(const std::string &_tune_name) { tune_name = _tune_name; }

      std::string getTuneName() const { return tune_name; }

      void setIterations(int _iterations) { iterations = _iterations; }

      Config *setArgs(const TypeArgTuple &t) {
        // arg_list<TypeArgs...>.push_back(t);
        arg_list.push_back(t);
        args_num++;
        return this;
      }

      int getArgsNum() const { return args_num; }

      int getIterations() const { return iterations; }

      template <typename Func> void loop_over_args_list(Func func) const {
        for (auto arg : arg_list) {
          func(iterations, arg);
        }
      }
    };
  } // namespace tuner
} // namespace usparse

#endif // USPARSE_TUNER_USPARSE_TUNER_CONFIG_HPP
