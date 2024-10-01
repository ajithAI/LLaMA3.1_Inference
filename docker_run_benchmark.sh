run="MaxFreq"
./docker_run_llama_inference.sh 64 2048 2048 10 50 ${run}
./docker_run_llama_inference.sh 96 2048 128 50 200 ${run}
./docker_run_llama_inference.sh 1024 128 128 25 100 ${run}
./docker_run_llama_inference.sh 1024 128 2048 5 25 ${run}

./docker_run_llama_inference.sh 64 2048 1 50 200 ${run}
./docker_run_llama_inference.sh 96 2048 1 50 200 ${run}
./docker_run_llama_inference.sh 1024 128 1 50 200 ${run}
