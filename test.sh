#!/bin/bash
#export VLLM_LOGGING_LEVEL=DEBUG
#export NCCL_DEBUG=TRACE
#export VLLM_TRACE_FUNCTION=1

outputlen=(6000 1500 1500 2000 1500 2000 1500 1000 1500)
numseqs=(1024 1024 16 128 512 128 512 512 512)
numprompts=(4000 2000 4000 4000 2000 2000 4000 4000 2000)
parallelsize=(1 2 4 4 4 4 8 8 8)
for i in {0..0}
do
        for j in {1..1}
        do
                for k in {2..2}
                do
                        for l in {0..0}
                        do
                                echo "****************************************"
                                python benchmarks/benchmark_sequence_inference.py \
                                --model /home/work05/Work/models/llm/Llama-2-7b-chat-hf \
                                --dataset /home/work02/work02.new/llm/benchmarks/vllmfile-main/data/ShareGPT_V3_unfiltered_cleaned_split.json \
                                --max-num-seqs ${numseqs[$k]} --output-len ${outputlen[$l]} --num-prompts ${numprompts[$j]} \
                                --tensor-parallel-size ${parallelsize[$i]} --sequence-parallel-size 2 \
                                --result /home/work02/work02.new/llm/vllm_sp/data/result.csv --enable-long-sequence 1\
                                --max-model-len 8192 --max-num-batched-tokens 1024\
                                --block-migrate-threshold 1024 --block-migrate-size 256 --block-migrate-start 512
                                echo "parallel size:${parallelsize[$i]},num prompts:${numprompts[$j]},num batched seqs:${numseqs[$k]},\
                                max output length:${outputlen[$l]}, sequence-parallel-size:2, max-model-len:8192, max-num-batched-tokens:1024"
                        done
                done
        done
done
