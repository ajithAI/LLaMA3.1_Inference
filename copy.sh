cp /home/user/LLaMA3.1_Inference/run_bm.py /home/user/LLaMA3.1_Inference/TensorRT-LLM/examples
cp /home/user/LLaMA3.1_Inference/utils.py /home/user/LLaMA3.1_Inference/TensorRT-LLM/examples
for i in {1..8}
do
cat dataset_llama_1024_2048_len.txt >> dataset_llama_8192_2048_len.txt
done
