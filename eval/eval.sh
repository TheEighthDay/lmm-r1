# qwen-max
python eval_zero_7b_acc.py --batch_size 4 --output_file results/china_qvq_max_eval_results.json --inference_engine doubao  --doubao_model_name=qvq-max  --doubao_api_key=xxxx --test_file=/data/phd/tiankaibin/dataset/data/china_street_view_data.jsonl & 


# doubao
python eval_zero_7b_acc.py --batch_size 4 --output_file results/china_doubao_eval_results.json --inference_engine doubao  --doubao_model_name=doubao-1.5-vision-pro-32k-250115  --doubao_api_key=xxx --test_file=/data/phd/tiankaibin/dataset/data/china_street_view_data.jsonl & 

#gpt4o
python eval_zero_7b_acc.py --batch_size 4 --output_file results/china_gpt4o_eval_results.json --inference_engine doubao  --doubao_model_name=gpt4o  --doubao_api_key=xx --test_file=/data/phd/tiankaibin/dataset/data/china_street_view_data.jsonl &

#gemini-2.0-flash-thinking-exp-01-21
python eval_zero_7b_acc.py --batch_size 4 --output_file results/china_gemini_2_think_pro_eval_results.json --inference_engine doubao --doubao_api_key=xxxx  --doubao_model_name=gemini-2.0-flash-thinking-exp-01-21 --test_file=/data/phd/tiankaibin/dataset/data/china_street_view_data.jsonl &

#gemini-2.0-pro-exp-02-05
python eval_zero_7b_acc.py --batch_size 4 --output_file results/china_gemini_2_pro_eval_results.json --inference_engine doubao --doubao_api_key=xxxx  --doubao_model_name="gemini-2.0-pro-exp-02-05" --test_file=/data/phd/tiankaibin/dataset/data/china_street_view_data.jsonl &

# qwen-7b cot
CUDA_VISIBLE_DEVICES=4 python eval_zero_7b_acc.py --batch_size 4 --output_file results/china_qwen7b_eval_results_cot_tkbnew.json --inference_engine vllm --model_name=Qwen/Qwen2.5-VL-7B-Instruct --cot --test_file=/data/phd/tiankaibin/dataset/data/china_street_view_data.jsonl &

# qwen-7b
CUDA_VISIBLE_DEVICES=5 python eval_zero_7b_acc.py --batch_size 4 --output_file results/china_qwen7b_eval_results.json --inference_engine vllm --model_name=Qwen/Qwen2.5-VL-7B-Instruct --test_file=/data/phd/tiankaibin/dataset/data/china_street_view_data.jsonl &

# qwen-32b
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_zero_7b_acc.py --batch_size 4 --output_file results/china_qwen32b_eval_results.json --inference_engine vllm --model_name=Qwen/Qwen2.5-VL-32B-Instruct --tensor_parallel --test_file=/data/phd/tiankaibin/dataset/data/china_street_view_data.jsonl &

# SFT
CUDA_VISIBLE_DEVICES=6 python eval_location_acc.py --model_name /data/phd/tiankaibin/lmm-r1/finetune/lora_merged_model --mode SFT --test_file /data/phd/tiankaibin/dataset/data/china_street_view_data.jsonl --inference_engine transformers  --output_file /data/phd/tiankaibin/lmm-r1/eval/china_lora_merged_model_SFT_eval_result.json &

# RL
CUDA_VISIBLE_DEVICES=7 python eval_location_acc.py --model_name /data/phd/tiankaibin/experiments_seekworld_system_length_kl0_2_easy3890_mid1882_hard5655/checkpoints/lmm-r1-seekworld-system-length-kl0_2_easy3890_mid1882_hard5655 --mode RL --test_file /data/phd/tiankaibin/dataset/data/china_street_view_data.jsonl --inference_engine vllm  --output_file /data/phd/tiankaibin/lmm-r1/eval/china_lmm-r1-seekworld_RL_eval_system_result_length_kl0_2_easy3890_mid1882_hard5655.json &


CUDA_VISIBLE_DEVICES=7 python eval_location_acc.py --model_name /data/phd/tiankaibin/experiments_seekworld_system2/checkpoints/lmm-r1-seekworld-system2 --mode RL --test_file /data/phd/tiankaibin/dataset/data/china_street_view_data.jsonl --inference_engine vllm  --output_file /data/phd/tiankaibin/lmm-r1/eval/china_lmm-r1-seekworld_RL_eval_system2.json &

