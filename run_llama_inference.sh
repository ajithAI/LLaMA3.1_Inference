BATCH=$1
ILEN=$2
OLEN=$3
WARM=$4
ITER=$5
EXT=$6

export DIR=/home/user/LLaMA3.1_Inference
mpirun -n 8 --allow-run-as-root --bind-to numa --rank-by hwthread --report-bindings python3 ${DIR}/TensorRT-LLM/examples/run_bm.py --run_profiling --tokenizer_dir=/home/user/Llama-3.1-70B --input_file=${DIR}/dataset_llama_8192_2048_len.txt --engine_dir=$DIR/TRT_Engines/LLaMA3.1_70B_TRT_Engine_FP8_8xGPU_MaxBatch_1024_MaxSeqLen_4096_CUDA_12.6_TRT_LLM_0.13_TP_8 --max_input_length=${ILEN} --max_output_len=${OLEN} --batch=${BATCH} --iterations=${ITER} --warmup=${WARM} 2>&1 | tee ${DIR}/${EXT}_RUN.txt
echo "LOGS SAVED AT : ${DIR}/${EXT}_RUN.txt"
