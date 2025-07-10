#!/bin/sh

# rm ls.log

ls *.tar.gz > ls.log

for f in $(cat ls.log); do
        tar -zxvf ${f} > /dev/null 2>&1
done

for f in $(ls ./); do
    dir_path=`pwd`
    dir_name="${dir_path}/${f}"
    if [ -d "${dir_name}" ]; then
        mv ${dir_name}/*.mtx ../data && rm -rf ${dir_name} > /dev/null 2>&1
    fi
done

rm ls.log