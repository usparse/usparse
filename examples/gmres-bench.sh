#!/bin/bash
#SBATCH -J gmres-bench
#SBATCH -o log-%j.log
#SBATCH -e log-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=dcu:1
#SBATCH -p hpctest06
#SBATCH -t 00:05:00
#SBATCH --exclusive
#SBATCH --export=ALL

MTX_PATH=/public/share/developers/sparse-dataset/suitesparse/bin2/
APP=./gmres-gpu-benchmark

$APP -m $MTX_PATH/100M/af_0_k101.bin2 -f bin2
$APP -m $MTX_PATH/100M/af_1_k101.bin2 -f bin2
$APP -m $MTX_PATH/100M/af_2_k101.bin2 -f bin2
$APP -m $MTX_PATH/100M/af_3_k101.bin2 -f bin2
$APP -m $MTX_PATH/100M/af_4_k101.bin2 -f bin2
$APP -m $MTX_PATH/100M/af_5_k101.bin2 -f bin2
$APP -m $MTX_PATH/100M/af_shell1.bin2 -f bin2
$APP -m $MTX_PATH/100M/af_shell2.bin2 -f bin2
$APP -m $MTX_PATH/100M/af_shell3.bin2 -f bin2
$APP -m $MTX_PATH/100M/af_shell4.bin2 -f bin2
$APP -m $MTX_PATH/100M/af_shell5.bin2 -f bin2
$APP -m $MTX_PATH/100M/af_shell6.bin2 -f bin2
$APP -m $MTX_PATH/100M/af_shell7.bin2 -f bin2
$APP -m $MTX_PATH/100M/af_shell8.bin2 -f bin2
$APP -m $MTX_PATH/100M/af_shell9.bin2 -f bin2
$APP -m $MTX_PATH/100M/af_shell10.bin2 -f bin2
$APP -m $MTX_PATH/1M/apache1.bin2 -f bin2
$APP -m $MTX_PATH/10M/apache2.bin2 -f bin2
$APP -m $MTX_PATH/10M/atmosmodd.bin2 -f bin2
$APP -m $MTX_PATH/10M/atmosmodj.bin2 -f bin2
$APP -m $MTX_PATH/100M/atmosmodl.bin2 -f bin2
$APP -m $MTX_PATH/100M/atmosmodm.bin2 -f bin2
$APP -m $MTX_PATH/100M/BenElechi1.bin2 -f bin2
$APP -m $MTX_PATH/10M/bmw7st_1.bin2 -f bin2
$APP -m $MTX_PATH/100M/bmwcra_1.bin2 -f bin2
$APP -m $MTX_PATH/100M/bone010.bin2 -f bin2
$APP -m $MTX_PATH/10M/boneS01.bin2 -f bin2
$APP -m $MTX_PATH/100M/boneS10.bin2 -f bin2
$APP -m $MTX_PATH/10M/cant.bin2 -f bin2
$APP -m $MTX_PATH/10M/cfd2.bin2 -f bin2
$APP -m $MTX_PATH/100M/circuit5M_dc.bin2 -f bin2
$APP -m $MTX_PATH/10M/CO.bin2 -f bin2
$APP -m $MTX_PATH/1M/cont-201.bin2 -f bin2
$APP -m $MTX_PATH/1M/cont-300.bin2 -f bin2
$APP -m $MTX_PATH/10M/cop20k_A.bin2 -f bin2
$APP -m $MTX_PATH/100M/crankseg_1.bin2 -f bin2
$APP -m $MTX_PATH/10M/crashbasis.bin2 -f bin2
$APP -m $MTX_PATH/10M/CurlCurl_1.bin2 -f bin2
$APP -m $MTX_PATH/10M/CurlCurl_2.bin2 -f bin2
$APP -m $MTX_PATH/100M/CurlCurl_3.bin2 -f bin2
$APP -m $MTX_PATH/10M/darcy003.bin2 -f bin2
$APP -m $MTX_PATH/10M/Dubcova3.bin2 -f bin2
$APP -m $MTX_PATH/10M/ecology1.bin2 -f bin2
$APP -m $MTX_PATH/10M/ecology2.bin2 -f bin2
$APP -m $MTX_PATH/1M/epb3.bin2 -f bin2
$APP -m $MTX_PATH/10M/F2.bin2 -f bin2
$APP -m $MTX_PATH/10M/FEM_3D_thermal2.bin2 -f bin2
$APP -m $MTX_PATH/10M/filter3D.bin2 -f bin2
$APP -m $MTX_PATH/1M/G2_circuit.bin2 -f bin2
$APP -m $MTX_PATH/10M/Ga3As3H12.bin2 -f bin2
$APP -m $MTX_PATH/10M/GaAsH6.bin2 -f bin2
$APP -m $MTX_PATH/10M/Goodwin_095.bin2 -f bin2
$APP -m $MTX_PATH/10M/Goodwin_127.bin2 -f bin2
$APP -m $MTX_PATH/10M/imagesensor.bin2 -f bin2
$APP -m $MTX_PATH/10M/laminar_duct3D.bin2 -f bin2
$APP -m $MTX_PATH/10M/majorbasis.bin2 -f bin2
$APP -m $MTX_PATH/10M/mario002.bin2 -f bin2
$APP -m $MTX_PATH/1M/matrix-new_3.bin2 -f bin2
$APP -m $MTX_PATH/10M/matrix_9.bin2 -f bin2
$APP -m $MTX_PATH/100M/ML_Laplace.bin2 -f bin2
$APP -m $MTX_PATH/10M/nd6k.bin2 -f bin2
$APP -m $MTX_PATH/100M/nd12k.bin2 -f bin2
$APP -m $MTX_PATH/100M/nd24k.bin2 -f bin2
$APP -m $MTX_PATH/10M/oilpan.bin2 -f bin2
$APP -m $MTX_PATH/10M/parabolic_fem.bin2 -f bin2
$APP -m $MTX_PATH/10M/poisson3Db.bin2 -f bin2
$APP -m $MTX_PATH/100M/SiO2.bin2 -f bin2
$APP -m $MTX_PATH/10M/sme3Dc.bin2 -f bin2
$APP -m $MTX_PATH/100M/ss.bin2 -f bin2
$APP -m $MTX_PATH/10M/stomach.bin2 -f bin2
$APP -m $MTX_PATH/10M/TEM152078.bin2 -f bin2
$APP -m $MTX_PATH/10M/TEM181302.bin2 -f bin2
$APP -m $MTX_PATH/10M/test1.bin2 -f bin2
$APP -m $MTX_PATH/10M/thermal2.bin2 -f bin2
$APP -m $MTX_PATH/10M/thermomech_dM.bin2 -f bin2
$APP -m $MTX_PATH/1M/thermomech_TC.bin2 -f bin2
$APP -m $MTX_PATH/1M/thermomech_TK.bin2 -f bin2
$APP -m $MTX_PATH/10M/tmt_sym.bin2 -f bin2
$APP -m $MTX_PATH/10M/tmt_unsym.bin2 -f bin2
$APP -m $MTX_PATH/10M/torso3.bin2 -f bin2
$APP -m $MTX_PATH/100M/Transport.bin2 -f bin2
