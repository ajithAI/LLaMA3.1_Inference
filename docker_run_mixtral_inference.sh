./run_clean.sh
sudo cpupower frequency-set -r -g performance
FOLDER=MIXTRAL_BM_LOGS
mkdir $FOLDER
FILE=${FOLDER}/Mixtral_8x7B_v0.1_TRT_Batch_${1}_Input_${2}_Output_${3}_${6}
turbostat -i 1 > ${FILE}_TURBOSTAT.txt &
nvidia-smi dmon -s pucvt > ${FILE}_NVIDIASMI.txt &
sleep 2s
docker exec TensorRT_LLM_8xGPU_CUDA_12.6.0 bash /home/user/Mixtral_Inference/run_mixtral_inference.sh ${1} ${2} ${3} ${4} ${5} ${FILE}
sleep 2s
pkill -9 turbostat
pkill -9 nvidia-smi
sleep 5s
./run_clean.sh
