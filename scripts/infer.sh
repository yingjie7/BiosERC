 conda activate env_py38/ && \
 python src/ft_llm.py  --do_eval_dev --do_eval_test  \
 --base_model_id "meta-llama/Llama-2-7b-hf"  --ft_model_id \
  "debug-check"  --lr_scheduler "linear" --lr "3e-4"   \
  --lora_r 32 --max_steps -1 --epoch 3  --kshot 0 --window 5 \
  --data_name "meld" --prompting_type "spdescV2" --extract_prompting_llm_id "Llama-2-70b-chat-hf" \
   --re_gen_data --seed 45  --max_seq_len 2048 --eval_delay 600 \
    --data_folder "./data"  \
    --ft_model_path "./finetuned_llm/meld_Llama-2-13b-hf_ep3_lrs-linear3e-4_0shot_r32_w5_spdescV2_devop_noise_s2048_R46_rc/"