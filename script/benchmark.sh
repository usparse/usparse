#!/usr/bin/bash
#SBATCH -J auto-test
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --exclusive
#SBATCH --gres=dcu:4

# source /public/home/wangxg/ustb/luxu/Set_env
# export HIP_KERNEL_PRINTF=1

input_dir=/public/home/wangxg/ustb/luxu/test/data
output_file=file.csv
exec_path=/public/home/wangxg/ustb/luxu/spgemm/build/bin/spgemm-hip
touch ${output_file}

# part1: matrix read from csr file
echo "csr-file, device-time, rocsparse-time, state" >> ${output_file}
for f in ${input_dir}/*; do
    base_f=`basename ${f}`
    tmplog=tmp-${base_f}.log
    ${exec_path} 2 ${f} > ${tmplog}
    dt=`grep "Device calculation finished!" ${tmplog} | grep -o 'time (mean):.*'`
    dt=${dt#*:}
    dt=${dt%(*}
    rt=`grep "Rocsparse calculation finished!" ${tmplog} | grep -o 'time:.*'`
    rt=${rt#*:}
    rt=${rt%(*}
    st=`grep "Congratulations, verification pass!" ${tmplog} | grep -o 'pass'`
    echo "${base_f}, ${dt}, ${rt}, ${st}" >> ${output_file}
done
    
