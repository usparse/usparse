#!/bin/sh
set -e
# this script move file directory and prepare for submitting standard.

tar_dir=cas-spgemm-ustb
solver_target_dir=src # directory to store spgemm source files in ${tar_dir}.
source_dir=../src/ # directory to copy files from
generated_dir=../build-hip/generated # directory to copy CMake generated files from.
git_info=1

if [ "$git_info" -eq 1 ]; then
    git_hash=$(git rev-parse --short HEAD)
    for crt_tag in $(git tag) # check git tag, if found, use git tag instead of git commit hash
    do
        if [ $(git rev-parse --short ${crt_tag}) == ${git_hash} ]; then
            git_hash=${crt_tag}
            break
        fi
    done
    tar_dir=${tar_dir}-${git_hash}
fi

echo "the soruce files will be sent in directory '${tar_dir}'."
mkdir -p ${tar_dir}
mkdir -p ${tar_dir}/${solver_target_dir}

# copy solver files.
cp -r ${source_dir}/spgemm ${tar_dir}/${solver_target_dir}
cp -r ${generated_dir} ${tar_dir}/${solver_target_dir}/
## remove CMakeLists.txt
find ${tar_dir}/${solver_target_dir} -name "CMakeLists.txt" -delete

# copy other files.
cp -r ${source_dir}/common_function.hpp ${tar_dir}
cp -r ${source_dir}/Csrsparse.hpp ${tar_dir}
cp -r ${source_dir}/main.cpp ${tar_dir}
cp -r ${source_dir}/Makefile ${tar_dir}
cp -r ./README-demo.md ${tar_dir}/README.md 
cp -r ${source_dir}/../.gitignore ${tar_dir}

## change Makefile
sed_args="-i \"\"" # macos
if [ $(uname) == 'Linux' ]; then
    sed -i "s|GENERATED_HEADER_PATH = ./generated|GENERATED_HEADER_PATH = ./${solver_target_dir}/generated|" ${tar_dir}/Makefile
    sed -i "s|SPGEMM_SOLVER_SRC_PATH = ./|SPGEMM_SOLVER_SRC_PATH = ./${solver_target_dir}|" ${tar_dir}/Makefile
else # mac/unix
    sed -i "" "s|GENERATED_HEADER_PATH = ./generated|GENERATED_HEADER_PATH = ./${solver_target_dir}/generated|" ${tar_dir}/Makefile
    sed -i "" "s|SPGEMM_SOLVER_SRC_PATH = ./|SPGEMM_SOLVER_SRC_PATH = ./${solver_target_dir}|" ${tar_dir}/Makefile
fi

tar zcvf ${tar_dir}.tar.gz ${tar_dir}
echo "file ${tar_dir}.tar.gz is generated"