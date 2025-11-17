#!/bin/bash

module purge
module load compiler/dtk/25.04 compiler/cmake/3.23.3
export CC=hipcc
export CXX=hipcc
