import json
import os
import torch
import re
import sys
import time
import argparse
from tqdm import tqdm
from datetime import datetime
from PIL import Image
import Levenshtein
from gpt4o_mini import make_request

# 导入调试工具
try:
    import ipdb
except ImportError:
    print("ipdb未安装，如需调试请安装: pip install ipdb")
    ipdb = None

# 导入模型相关库
from transformers import AutoProcessor
# 条件导入，根据选择的推理引擎
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    transformers_available = True
except ImportError:
    transformers_available = False

try:
    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info
    vllm_available = True
except ImportError:
    vllm_available = False

# 重新实现验证函数
def standardize_region_name(region):
    """
    标准化地区名称，移除常见的后缀
    """
    region = region.lower().strip()
    # 移除常见的后缀
    suffixes = [
        ' province', ' sheng', ' län', ' oblast', ' governorate',' city',
        ' province', ' state', ' territory', ' region', ' district'
    ]
    for suffix in suffixes:
        if region.endswith(suffix):
            region = region[:-len(suffix)]
    return region.strip()

def compare_region_words(pred_words, true_words):
    """
    比较两个地区名称的单词是否匹配
    使用 Levenshtein 距离，如果两个单词的距离小于3，则认为匹配(有些地区因为发音不一样，存在1-2个字符差异)
    """
    if not pred_words or not true_words:
        return False
        
    # 记录已匹配的单词，避免重复匹配
    matched_true_words = set()
    
    # 对每个预测的单词
    for pred_word in pred_words:
        found_match = False
        # 在真实单词中寻找匹配
        for i, true_word in enumerate(true_words):
            if i in matched_true_words:
                continue
            # 如果 Levenshtein 距离小于3，认为匹配
            if Levenshtein.distance(pred_word, true_word) < 3:
                matched_true_words.add(i)
                found_match = True
                break
        # 如果当前预测单词没有找到匹配，返回False
        if not found_match:
            return False
            
    # 检查是否所有真实单词都被匹配
    return len(matched_true_words) == len(true_words)

def verify_location_with_gpt4o(pred_answer, true_answer):
    answer_match = re.search(r"<answer>(.*?)</answer>", pred_answer)
    if answer_match:
        pred_answer = answer_match.group(1).strip()
    else:
        # 匹配失败时使用原始文本
        pred_answer = pred_answer.strip()
    # 如果不在列表中，使用 GPT4O 进行验证
    prompt = f"""请判断以下两个位置是否指的是同一个地方，要求1. 国家（country）2. 行政省份/州（administrative_area_level_1），完全一致才算同一个地方，如果administrative_area_level_1存在多个别名，用/分隔，只要有一个正确就行：
预测位置：{pred_answer}
真实位置：{true_answer}

分析原因，最后回答 [是] 或 [否]。 注意[是] 和 [否] 是用[]括起来的。"""
    
    try:
        response = make_request(prompt=prompt)
        answer = response['response'].strip()
        return 1.0 if "[是]" in answer and "[否]" not in answer else 0.0
    except Exception as e:
        print(f"GPT4O 验证出错: {str(e)}")
        return 0.0

def verify_location(content, sol):
    """
    验证位置信息的正确性
    Args:
        content: 模型的输出，格式为 <think>...</think><answer>$country,region$</answer>
        sol: 真实答案，格式为 $country,region1/region2/region3$，其中 region1/region2/region3 是同一个地区的不同别名
    Returns:
        float: 1.0 如果完全匹配，0.0 如果国家匹配但地区不匹配，0.0 如果国家不匹配
    """
    # 提取答案部分
    answer_match1 = re.search(r"<answer>\$(.*?)\$</answer>", content)
    answer_match2 = re.search(r"\$(.*?)\$", content)
    answer_match = None

    if answer_match1:
        answer_match = answer_match1
    elif answer_match2:
        answer_match = answer_match2

    if not answer_match:
        return 0.0
    
    # 清理答案中的空格
    answer = answer_match.group(1).strip()
    sol = sol.strip().strip('$').strip()
    
    # 分割国家和行政区
    try:
        pred_country, pred_region = [x.strip().lower() for x in answer.split(',')]
        true_country, true_regions = [x.strip().lower() for x in sol.split(',')]
    except ValueError:
        return 0.0
    
    # 国家必须完全匹配
    if pred_country != true_country:
        return 0.0
    
    # 标准化预测的地区名称
    if "/" in pred_region:
        pred_region = pred_region.split("/")[0]
    pred_region = standardize_region_name(pred_region)

    
    # 将预测的地区名称分割成单词
    pred_words = [x.strip() for x in pred_region.split()]
    
    # 检查预测的地区是否匹配任何一个真实地区别名
    for true_region in true_regions.split('/'):
        true_region = standardize_region_name(true_region)
        true_words = [x.strip() for x in true_region.split()]
        
        # 如果单词匹配，返回1.0
        if compare_region_words(pred_words, true_words):
            return 1.0
    
    return 0.0

def setup_logger(output_prefix=None):
    """设置日志记录器"""
    # 创建logs目录
    os.makedirs('logs', exist_ok=True)
    
    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_prefix:
        log_file = f'logs/{output_prefix}_location_acc.log'
    else:
        log_file = f'logs/eval_location_acc_{timestamp}.log'
    
    # 创建日志文件
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"评估开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    return log_file

def log_result(log_file, image_path, prediction, ground_truth, score, raw_data, error_type=None):
    """记录分析结果"""
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"图片路径: {image_path}\n")
        f.write(f"预测结果: {prediction}\n")
        f.write(f"真实结果: {ground_truth}\n")
        f.write(f"是否匹配: {'是' if score == 1.0 else '否'}\n")
        if error_type:
            f.write(f"错误类型: {error_type}\n")
        f.write("原始数据:\n")
        f.write(json.dumps(raw_data, ensure_ascii=False, indent=2))
        f.write("\n" + "="*50 + "\n\n")

def load_jsonl(file_path):
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def extract_answer(response):
    """从响应中提取答案部分"""
    answer_match = re.search(r"<answer>\$(.*?)\$</answer>", response)
    if answer_match:
        return answer_match.group(1).strip()
    return None

def evaluate_location_accuracy(model_name, mode, test_file=None, inference_engine="vllm", debug=False, batch_size=4, user_cot=False, output_prefix=None, output_file=None):
    """评估位置识别准确率"""
    # 设置日志
    log_file = setup_logger(output_prefix)
    
    # 加载模型和处理器
    print(f"加载模型: {model_name}")
    print(f"推理引擎: {inference_engine}")
    print(f"批处理大小: {batch_size}")
    
    # 根据选择的推理引擎加载模型
    processor = AutoProcessor.from_pretrained(model_name, padding_side='left')
    
    if inference_engine == "vllm":
        if not vllm_available:
            raise ImportError("vLLM 库不可用，请安装 vllm 或选择 transformers 引擎")
        
        # 使用vLLM加载模型
        llm = LLM(
            model=model_name,
            limit_mm_per_prompt={"image": 10, "video": 10},
            dtype="auto",
            gpu_memory_utilization=0.95,
        )
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.05,
            max_tokens=512,
            stop_token_ids=[],
        )
    else:  # transformers
        if not transformers_available:
            raise ImportError("Transformers 相关库不可用，请安装必要的包")
        
        # 使用transformers加载模型
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="cuda:0"
        )

    print(f"加载测试数据: {test_file}")
    data = load_jsonl(test_file)
    
    # 输出示例数据，帮助调试
    if debug and len(data) > 0:
        print("\n=== 数据样例 ===")
        print(f"数据类型: {type(data)}")
        print(f"样本数量: {len(data)}")
        example = data[0]
        print(f"示例数据类型: {type(example)}")
        print(f"示例数据字段: {list(example.keys())}")
        for key, value in example.items():
            print(f"字段 '{key}' 类型: {type(value)}")
            if isinstance(value, str) and key == 'message':
                print(f"字段 '{key}' 前100个字符: {value[:100]}...")
        print("=== 样例结束 ===\n")
    
    # 统计结果
    total = len(data)
    correct = 0
    wrong = 0
    
    # 记录所有案例
    all_cases = []
    
    # 批量处理
    for i in tqdm(range(0, len(data), batch_size), desc="评估进度"):
        batch_data = data[i:i + batch_size]
        batch_messages = []
        batch_image_paths = []
        batch_ground_truths = []
        batch_items = []
        
        if inference_engine == "vllm":
            batch_inputs = []
        
        # 准备批次数据
        for item in batch_data:
            # 提取用户消息
            if debug:
                ipdb.set_trace()
            if True:
                messages = json.loads(item['message'])
                system_message = messages[0]
                user_message = messages[1]
                ground_truth = item['answer']
                
                # 提取图像路径和真实答案
                image_path = None
                for content in user_message['content']:
                    if content['type'] == 'image':
                        image_path = content['image']
                
                if not image_path or not ground_truth:
                    print(f"跳过不完整数据: {item}")
                    continue
                
                # 确认图像文件存在
                if not os.path.exists(image_path):
                    print(f"图像文件不存在: {image_path}")
                    wrong += 1
                    all_cases.append({
                        'image': image_path,
                        'prediction': '图像文件不存在',
                        'ground_truth': ground_truth,
                        'is_correct': False,
                        'error_type': 'file_not_found'
                    })
                    # 记录结果
                    log_result(log_file, image_path, '图像文件不存在', ground_truth, 0.0, item, 'file_not_found')
                    continue
                
                # 打开图像
                try:
                    image = Image.open(image_path)
                except Exception as e:
                    print(f"加载图像失败: {image_path}, 错误: {str(e)}")
                    wrong += 1
                    all_cases.append({
                        'image': image_path,
                        'prediction': f'加载图像失败: {str(e)}',
                        'ground_truth': ground_truth,
                        'is_correct': False,
                        'error_type': 'image_load_error'
                    })
                    # 记录结果
                    log_result(log_file, image_path, f'加载图像失败: {str(e)}', ground_truth, 0.0, item, 'image_load_error')
                    continue
                
                # 构建处理消息

                if mode == "RL":
                    prompt_messages = [
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": system_message["content"]}
                            ]
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": user_message["content"][1]["text"]}
                            ]
                        }
                    ]
                elif mode == "SFT":
                    prompt_messages = [
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": "You are a helpful assistant good at locating the country and first-level administrative region of a picture."}
                            ]
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": user_message["content"][1]["text"]}
                            ]
                        }
                    ]
                else:
                    if user_cot:
                        
                        prompt_messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": image},
                                    {"type": "text", "text": system_message["content"]+user_message["content"][1]["text"]}
                                ]
                            }
                        ]
                    else:
                        cot = "Question1:Arethereprominent natural features, such as specific types of vegetation, landforms (e.g., mountains, hills, plains), or soil characteristics, that provide clues about the geographicalregion? • Question2: Are there any culturally, historically, or architecturally significant landmarks, buildings, or structures, or are there any inscriptions or signs in a specific language or script that could help determine the country or region? • Question3: Are there distinctive road-related features, such as traffic direction (e.g., left-hand or righthand driving), specific types of bollards, unique utility pole designs, or license platecolors and styles, which countries are known to have these characteristics? • Question4: Are there observable urban or rural markers (e.g., street signs, fire hydrants guideposts) , or other infrastructure elements, that can provide more specific information about the country or city? • Question5: Are there identifiable patterns in sidewalks (e.g., tile shapes, colors, or arrangements), clothing styles worn by people, or other culturally specific details that can help narrow down the city or area? Let's think step by step. Based on the question I provided, locate the location of the picture as accurately as possible."
                        prompt_messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": image},
                                    {"type": "text", "text": system_message["content"] + cot+user_message["content"][1]["text"]}
                                ]
                            }
                        ]

                        
                
                # vLLM特有的处理
                if inference_engine == "vllm":
                    # 处理消息为vLLM格式
                    prompt = processor.apply_chat_template(
                        prompt_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    
                    # 处理图像数据
                    image_inputs, video_inputs = process_vision_info(prompt_messages)
                    
                    mm_data = {}
                    if image_inputs is not None:
                        mm_data["image"] = image_inputs
                    
                    # 构建vLLM输入
                    llm_input = {
                        "prompt": prompt,
                        "multi_modal_data": mm_data,
                    }
                    batch_inputs.append(llm_input)
                
                batch_messages.append(prompt_messages)
                batch_image_paths.append(image_path)
                batch_ground_truths.append(ground_truth)
                batch_items.append(item)
                
            # except Exception as e:
            #     print(f"处理数据项时出错: {str(e)}")
            #     continue
        
        if not batch_messages:
            continue
        
        # 批量生成回答
        try:
            responses = []
            
            if inference_engine == "vllm":
                # 使用vLLM进行推理
                outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
                
                # 提取生成的文本
                for output in outputs:
                    responses.append(output.outputs[0].text)
                    
            else:  # transformers
                # 准备输入
                texts = [processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                ) for messages in batch_messages]
                
                # 进行推理
                if mode == "SFT" or mode == "RL":
                    inputs = processor(
                        text=texts,
                        images=[messages[1]['content'][0]['image'] for messages in batch_messages],
                        return_tensors="pt",
                        padding=True,
                    )
                else:
                    inputs = processor(
                        text=texts,
                        images=[messages[0]['content'][0]['image'] for messages in batch_messages],
                        return_tensors="pt",
                        padding=True,
                    )
                

                inputs = inputs.to(model.device)
                # 生成回答
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=512)
                
 
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
                ]
                
                responses = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            
            # 处理每个回答（统一处理逻辑）
            for response, image_path, ground_truth, item in zip(responses, batch_image_paths, batch_ground_truths, batch_items):
                print("__________________________")
                print(response)
                print("__________________________")
                # 构建用于验证的完整响应
                if mode == "SFT":
                    verification_content = "<answer>" + response + "</answer>"
                else:
                    verification_content = response
                
                # 使用验证函数
                score = verify_location_with_gpt4o(verification_content, ground_truth)
                
                # 记录结果
                error_type = None if score == 1.0 else 'location_mismatch'
                log_result(log_file, image_path, response, ground_truth, score, item, error_type)
                
                # 统计结果
                if score == 1.0:
                    correct += 1
                    all_cases.append({
                        'image': image_path,
                        'prediction': response,
                        'ground_truth': ground_truth,
                        'is_correct': True,
                        'error_type': None
                    })
                else:
                    wrong += 1
                    all_cases.append({
                        'image': image_path,
                        'prediction': response,
                        'ground_truth': ground_truth,
                        'is_correct': False,
                        'error_type': 'location_mismatch'
                    })
                
        except Exception as e:
            print(f"模型推理失败: {str(e)}")
            for image_path, ground_truth, item in zip(batch_image_paths, batch_ground_truths, batch_items):
                wrong += 1
                all_cases.append({
                    'image': image_path,
                    'prediction': f'模型推理失败: {str(e)}',
                    'ground_truth': ground_truth,
                    'is_correct': False,
                    'error_type': 'model_inference_error'
                })
                # 记录结果
                log_result(log_file, image_path, f'模型推理失败: {str(e)}', ground_truth, 0.0, item, 'model_inference_error')
        
        # 避免请求过快
        time.sleep(0.1)
        
        # 清理GPU缓存，避免内存泄漏
        if inference_engine == "transformers" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 计算准确率
    accuracy = correct / total if total > 0 else 0
    
    # 打印统计结果
    print(f"\n总样本数: {total}")
    print(f"正确样本数: {correct} ({accuracy:.2%})")
    print(f"错误样本数: {wrong} ({(1-accuracy):.2%})")
    
    # 保存所有案例
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_prefix:
        cases_file = f'logs/{output_prefix}_cases.json'
    else:
        cases_file = f'logs/eval_cases_{timestamp}.json'
    
    with open(cases_file, 'w', encoding='utf-8') as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)
    print(f"\n所有案例已保存到: {cases_file}")
    
    # 记录评估结束时间
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n评估结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总样本数: {total}\n")
        f.write(f"正确样本数: {correct} ({accuracy:.2%})\n")
        f.write(f"错误样本数: {wrong} ({(1-accuracy):.2%})\n")

    # 如果指定了输出文件，保存结果
    if output_file:
        output_data = {
            "accuracy": float(accuracy),
            "total_samples": total,
            "correct_samples": correct,
            "wrong_samples": wrong,
            "model_name": model_name,
            "mode": mode,
            "inference_engine": inference_engine,
            "test_file": test_file,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "detailed_results": all_cases
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # 保存到输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n结果详情已保存到: {output_file}")

    return accuracy

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='评估模型位置识别准确率')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help='模型名称或路径')
    parser.add_argument('--mode', type=str, default="SFT", choices=["SFT", "RL"],
                        help='验证模式: SFT 或 default')
    parser.add_argument('--test_file', type=str, default="qwen_format_v3/test.jsonl",
                        help='测试数据文件路径')
    parser.add_argument('--inference_engine', type=str, default="vllm", choices=["vllm", "transformers"],
                        help='推理引擎: vllm 或 transformers')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批处理大小，较小的值可减少内存使用')
    parser.add_argument('--user_cot', action='store_true',
                        help='使用用户COT')
    parser.add_argument('--output_prefix', type=str, default=None,
                        help='结果文件名前缀，用于日志和案例文件的命名')
    parser.add_argument('--output_file', type=str, default=None,
                        help='结果文件路径')
    
    args = parser.parse_args()
    
    # 如果指定了设备，设置默认设备
            
    # 对于transformers引擎，设置内存使用率以避免OOM错误
    if args.inference_engine == "transformers" and torch.cuda.is_available():
        # 为每个GPU设置内存使用比例为0.8
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(0.8, i)
            print(f"已为GPU {i} 设置内存使用比例为0.6")
    
    print("\n开始位置识别准确率评估...")
    print(f"模型: {args.model_name}")
    print(f"模式: {args.mode}")
    print(f"推理引擎: {args.inference_engine}")
    print(f"调试模式: {'启用' if args.debug else '禁用'}")
    if args.test_file:
        print(f"测试文件: {args.test_file}")
    
    accuracy = evaluate_location_accuracy(
        model_name=args.model_name,
        mode=args.mode,
        test_file=args.test_file,
        inference_engine=args.inference_engine,
        debug=args.debug,
        batch_size=args.batch_size,
        user_cot=args.user_cot,
        output_prefix=args.output_prefix,
        output_file=args.output_file
    )
    print(f"最终准确率: {accuracy:.2%}")


# CUDA_VISIBLE_DEVICES=2 python eval_location_acc.py --model_name /data/phd/tiankaibin/lmm-r1/finetune/lora_merged_model --mode SFT --test_file /data/phd/tiankaibin/dataset/data/test.jsonl --inference_engine transformers  --output_file /data/phd/tiankaibin/lmm-r1/eval/lora_merged_model_SFT_eval_result.json

# CUDA_VISIBLE_DEVICES=0 python eval_location_acc.py --model_name /data/phd/tiankaibin/experiments_seekworld_system2/checkpoints/lmm-r1-seekworld-system2 --mode RL --test_file /data/phd/tiankaibin/dataset/data/test.jsonl --inference_engine transformers  --output_file /data/phd/tiankaibin/lmm-r1/eval/lmm-r1-seekworld_RL_eval_system_result.json

# CUDA_VISIBLE_DEVICES=1 python eval_location_acc_nosystem.py --model_name /data/phd/tiankaibin/experiments_seekworld_system2/checkpoints/lmm-r1-seekworld-system2 --mode RL --test_file /data/phd/tiankaibin/dataset/data/test.jsonl --inference_engine transformers  --output_file /data/phd/tiankaibin/lmm-r1/eval/lmm-r1-seekworld_RL_eval_nosystem_result.json

# CUDA_VISIBLE_DEVICES=0 python eval_location_acc_nosystem.py --model_name /data/phd/tiankaibin/experiments_seekworld/checkpoints/lmm-r1-seekworld/ckpt/global_step120_hf --mode RL --test_file /data/phd/tiankaibin/dataset/data/test.jsonl --inference_engine vllm  --output_file /data/phd/tiankaibin/lmm-r1/eval/lmm-r1-seekworld_RL_eval_result_nosystem.json


# CUDA_VISIBLE_DEVICES=0 python eval_location_acc.py --model_name /data/phd/tiankaibin/experiments_seekworld_kl_0_train_easy1945_mid941/checkpoints/lmm-r1-seekworld-kl_0_train_easy1945_mid941 --mode RL --test_file /data/phd/tiankaibin/dataset/data/test.jsonl --inference_engine vllm  --output_file /data/phd/tiankaibin/lmm-r1/eval/lmm-r1-seekworld-kl_0_train_easy1945_mid941_RL_eval_result.json

# CUDA_VISIBLE_DEVICES=1 python eval_location_acc.py --model_name /data/phd/tiankaibin/experiments_seekworld_kl_0_train_mid941_hard941/checkpoints/lmm-r1-seekworld-kl_0_train_mid941_hard941 --mode RL --test_file /data/phd/tiankaibin/dataset/data/test.jsonl --inference_engine vllm  --output_file /data/phd/tiankaibin/lmm-r1/eval/lmm-r1-seekworld-kl_0_train_mid941_hard941_RL_eval_result.json

# CUDA_VISIBLE_DEVICES=2 python eval_location_acc.py --model_name /data/phd/tiankaibin/experiments_seekworld_kl_0_train_easy1945_mid941_hard2886/checkpoints/lmm-r1-seekworld-kl_0_train_easy1945_mid941_hard2886 --mode RL --test_file /data/phd/tiankaibin/dataset/data/test.jsonl --inference_engine vllm  --output_file /data/phd/tiankaibin/lmm-r1/eval/lmm-r1-seekworld-kl_0_train_easy1945_mid941_hard2886_RL_eval_result.json