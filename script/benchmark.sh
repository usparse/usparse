#!/usr/bin/bash
#SBATCH -J auto-test
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --exclusive
#SBATCH --gres=dcu:4

# source /public/home/wangxg/ustb/luxu/Set_env
# export HIP_KERNEL_PRINTF=1

input_dir=/public/home/wangxg/ustb/luxu/suitesparse-data/10k
output_file=dcu-419-10k.csv
exec_path=/public/home/wangxg/ustb/luxu/spgemm/build/bin/spgemm-benchmark
touch ${output_file}

# part1: matrix read from csr file
echo "csr-file, rows, cols, nnz, USTB-SpGEMM-Total, USTB-SpGEMM-Symbolic, USTB-SpGEMM-Numeric, USTB-SpGEMM-nnz, Rocsparse-Total, Rocsparse-Symbolic, Rocsparse-Numeric, Rocsparse-nnz, state" >> ${output_file}
for sub_dir in $(ls ${input_dir}); do
    for f in ${input_dir}/${sub_dir}/*; do
        if [ -f "${f}" ]; then
            base_f=`basename ${f}`
            tmplog=tmp-${base_f}.log
            ${exec_path} ${f} 1 "bin"> ${tmplog}

            rows=`grep "rows of matrix_in" ${tmplog} | grep -o '=.*'`
            rows=${rows#*=}

            cols=`grep "cols of matrix_in" ${tmplog} | grep -o '=.*'`
            cols=${cols#*=}

            nnz=`grep "nnz of matrix_in" ${tmplog} | grep -o '=.*'`
            nnz=${nnz#*=}

            ustbtotal=`grep "USTB-SpGEMM calculation finished! Total elapsed time (mean)" ${tmplog} | grep -o 'time (mean):.*'`
            ustbtotal=${ustbtotal#*:}
            ustbtotal=${ustbtotal%(*}

            ustbsymbolic=`grep "USTB-SpGEMM calculation finished! Symbolic elapsed time (mean)" ${tmplog} | grep -o 'time (mean):.*'`
            ustbsymbolic=${ustbsymbolic#*:}
            ustbsymbolic=${ustbsymbolic%(*}

            ustbnumeric=`grep "USTB-SpGEMM calculation finished! Numeric elapsed time (mean)" ${tmplog} | grep -o 'time (mean):.*'`
            ustbnumeric=${ustbnumeric#*:}
            ustbnumeric=${ustbnumeric%(*}

            ustbnnz=`grep "USTB-SpGEMM calculate non zero num of matrix C" ${tmplog} | grep -o 'matrix C:.*'`
            ustbnnz=${ustbnnz#*:}

            roctotal=`grep "Rocsparse-SpGEMM calculation finished! Total elapsed time (mean)" ${tmplog} | grep -o 'time (mean):.*'`
            roctotal=${roctotal#*:}
            roctotal=${roctotal%(*}

            rocsymbolic=`grep "Rocsparse-SpGEMM calculation finished! Symbolic elapsed time (mean)" ${tmplog} | grep -o 'time (mean):.*'`
            rocsymbolic=${rocsymbolic#*:}
            rocsymbolic=${rocsymbolic%(*}

            rocnumeric=`grep "Rocsparse-SpGEMM calculation finished! Numeric elapsed time (mean)" ${tmplog} | grep -o 'time (mean):.*'`
            rocnumeric=${rocnumeric#*:}
            rocnumeric=${rocnumeric%(*}

            rocnnz=`grep "Rocsparse-SpGEMM calculate non zero num of matrix C" ${tmplog} | grep -o 'matrix C:.*'`
            rocnnz=${rocnnz#*:}

            state=`grep "Congratulations, verification pass!" ${tmplog} | grep -o 'pass'`
            
            echo "${base_f}, ${rows}, ${cols}, ${nnz}, ${ustbtotal}, ${ustbsymbolic}, ${ustbnumeric}, ${ustbnnz}, ${roctotal}, ${rocsymbolic}, ${rocnumeric}, ${rocnnz}, ${state}" >> ${output_file}
        fi
    done
done
    
