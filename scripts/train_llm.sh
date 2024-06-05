#!/usr/bin/bash
#
#         Job Script for VPCC , JAIST
#                                    2018.2.25 

#PBS -N erc-llm
#PBS -j oe 
#PBS -q GPU-1
#PBS -o pbs_infer-sp.log
#PBS -e infer-sp.err.log 

source ~/.bashrc

conda activate ~/per_erc/env_llm
cd ~/per_erc/

EP=3
LR_SCHEDULER="linear"
LR=3e-4
LORA_R=32
TOPK=0
WINDOW=5
PROMPT_TYPE="default"
MODEL_ID="meta-llama/Llama-2-7b-hf"   #  "meta-llama/Llama-2-7b-hf" # codellama/CodeLlama-7b-hf
DATANAME="iemocap"   #  "meta-llama/Llama-2-7b-hf" # codellama/CodeLlama-7b-hf
EXTRACT_PROMTING_LLM_ID="Llama-2-70b-chat-hf" #Meta-Llama-3-8B-Instruct
MAX_SEQ_LEN=1024
MAX_STEPS=-1
EVAL_DELAY=600

# # FT_MODEL_PATH="/home/phuongnm/SP_LLMs/finetuned_llm/lora_ep10_lrs-linear_cotv2"

IFS='/' read -ra ADDR <<< "$MODEL_ID"
MODEL_ID_0=${ADDR[1]}

for seed in 42 43 44 45 46 ;
do 
python /home/s2220429/per_erc/src/ft_llm.py  --do_eval_dev --do_eval_test --do_train \
 --base_model_id $MODEL_ID \
 --ft_model_id  ${DATANAME}_${MODEL_ID_0}_ep${EP}_step${MAX_STEPS}_lrs-${LR_SCHEDULER}${LR}_${TOPK}shot_r${LORA_R}_w${WINDOW}_${PROMPT_TYPE}_seed${seed}_L${MAX_SEQ_LEN}_llmdesc${EXTRACT_PROMTING_LLM_ID}_ED${EVAL_DELAY} \
 --lr_scheduler $LR_SCHEDULER --lr $LR   --lora_r $LORA_R --max_steps $MAX_STEPS --epoch ${EP} \
 --kshot $TOPK --window $WINDOW --data_name $DATANAME --prompting_type ${PROMPT_TYPE} --extract_prompting_llm_id $EXTRACT_PROMTING_LLM_ID \
 --re_gen_data --seed $seed  --max_seq_len $MAX_SEQ_LEN --eval_delay $EVAL_DELAY \
#  --ft_model_path="/home/s2220429/per_erc/finetuned_llm/meld_Llama-2-7b-hf_ep3_lrs-linear3e-4_0shot_r32_w5_multitask_step1_seed${seed}_L2048_llmdescLlama-2-70b-chat-hf/" \


done

wait

