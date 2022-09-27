#!/bin/sh

set -e

path=$1
mkdir -p $path

# configs:
src_base="../src/"
generated_file="../build-hip/generated/building_config.h"

# the file order is very important.
source_files=(
    "${generated_file}"
    "${src_base}/spgemm/api/spgemm_hip_types.h"
    "${src_base}/spgemm/api/spgemm_hip_api.h"
    "${src_base}/spgemm/common/check/hip_check.hpp"
    "${src_base}/spgemm/common/scan/scan_in_device.hpp"
    "${src_base}/spgemm/common/scan/single_block_scan.hpp"
    "${src_base}/spgemm/common/scan/reduce_then_scan.hpp"
    "${src_base}/spgemm/common/scan/lookback_scan.hpp"
    "${src_base}/spgemm/common/scan/scan_impl.hpp"
    "${src_base}/spgemm/common/max/block_reduce_max.hpp"
    "${src_base}/spgemm/common/max/max_impl.hpp"
    "${src_base}/spgemm/common/timer/timer.h"
    "${src_base}/spgemm/common/common.h"
    "${src_base}/spgemm/hip-kokkos/align.h"
    "${src_base}/spgemm/hip-kokkos/compress_matrix.h"
    "${src_base}/spgemm/hip-kokkos/csr_matrix.h"
    "${src_base}/spgemm/hip-kokkos/hash_map.h"
    "${src_base}/spgemm/hip-kokkos/hash_map_insert.hpp"
    "${src_base}/spgemm/hip-kokkos/spgemm_compress_kernel.hpp"
    "${src_base}/spgemm/hip-kokkos/spgemm_compress_impl.hpp"
    "${src_base}/spgemm/hip-kokkos/spgemm_symbolic_kernel.hpp"
    "${src_base}/spgemm/hip-kokkos/spgemm_symbolic_impl.hpp"
    "${src_base}/spgemm/hip-kokkos/split_group_kernel.hpp"
    "${src_base}/spgemm/hip-kokkos/split_config.h"
    "${src_base}/spgemm/hip-kokkos/split_group_impl.hpp"
    "${src_base}/spgemm/hip-kokkos/spgemm_numeric_kernel.hpp"
    "${src_base}/spgemm/hip-kokkos/spgemm_numeric_impl.hpp"
    "${src_base}/spgemm/hip-kokkos/spgemm_core_impl.hpp"
    "${src_base}/spgemm/hip-kokkos/spgemm.h"
    "${src_base}/spgemm/api/spgemm_hip_api.cc"
)

header_only_file="$path/header-only.hpp"
touch $header_only_file
echo "" > $header_only_file # clean file
for f in ${source_files[@]}
do
    cat $f >> $header_only_file
    echo "" >> $header_only_file
done

cat ${src_base}/Csrsparse.hpp >> $header_only_file

# copy demo files.
tar_dir="$path/"
cp -r ${src_base}/common_function.hpp ${tar_dir}
cp -r ${src_base}/main.cpp ${tar_dir}
cp -r ${src_base}/../.gitignore ${tar_dir}
cp -r ./Makefile-demo ${tar_dir}/Makefile
cp -r ./README-demo.md ${tar_dir}/README.md

# process final source file and delete the origin header-only file:
sed '/^#include ".*/d' $header_only_file > ${tar_dir}/Csrsparse.hpp
rm -rf $header_only_file

echo "done, files stored at '$tar_dir'"
