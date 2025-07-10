#!/usr/bin/bash
#SBATCH -J auto-test
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --exclusive
#SBATCH --gres=dcu:4

# source /public/home/wangxg/ustb/luxu/Set_env
# export HIP_KERNEL_PRINTF=1

input_dir=/public/home/wangxg/ustb/luxu/data
output_file=file.csv
exec_path=/public/home/wangxg/ustb/luxu/spgemm/build/bin/spgemm-hip
rates=(0.001 0.01 0.02)
seed=20
touch ${output_file}

# part1: matrix read from csr file
echo "csr-file, sparseness, seed, device-time, rocsparse-time, state" >> ${output_file}
for f in ${input_dir}/*; do
    for rate in ${rates[@]}; do
        base_f=`basename ${f}`
        tmplog=tmp-${base_f}-${rate}-${seed}.log
        ${exec_path} 1 ${f} ${rate} ${seed} > ${tmplog} 
        dt=`grep "Device calculation finished!" ${tmplog} | grep -o 'time (mean):.*'`
        dt=${dt#*:}
        dt=${dt%(*}
        rt=`grep "Rocsparse calculation finished!" ${tmplog} | grep -o 'time:.*'`
        rt=${rt#*:}
        rt=${rt%(*}
        st=`grep "Congratulations, verification pass!" ${tmplog} | grep -o 'pass'`
        echo "${base_f}, ${rate}, ${seed}, ${dt}, ${rt}, ${st}" >> ${output_file}
    done
done

# part2: matrix generate by random
rates=(0.0001 0.001 0.002)
for rate in ${rates[@]}; do
    tmplog=tmp-TEST-${rate}-${seed}.log
    ${exec_path} 0 50000 50000 0.00007 ${rate} ${seed} > ${tmplog}
    dt=`grep "Device calculation finished!" ${tmplog} | grep -o 'time (mean):.*'`
    dt=${dt#*:}
    dt=${dt%(*}
    rt=`grep "Rocsparse calculation finished!" ${tmplog} | grep -o 'time:.*'`
    rt=${rt#*:}
    rt=${rt%(*}
    st=`grep "Congratulations, verification pass!" ${tmplog} | grep -o 'pass'`
    echo "TEST, ${rate}, ${seed}, ${dt}, ${rt}, ${st}" >> ${output_file}
done
    
