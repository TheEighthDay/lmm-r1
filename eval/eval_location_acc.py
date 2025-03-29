import json
import os
import torch
import re
import sys
import time
from tqdm import tqdm
from datetime import datetime
from PIL import Image
import Levenshtein

# 导入模型相关
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# 删除外部导入
# sys.path.append('/data/phd/tiankaibin/lmm-r1/openrlhf/models/remote_rm')
# from location_verifier import standardize_region_name, compare_region_words, verify_location

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
    answer_match = re.search(r"<answer>\$(.*?)\$</answer>", content)
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

def setup_logger():
    """设置日志记录器"""
    # 创建logs目录
    os.makedirs('logs', exist_ok=True)
    
    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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

def evaluate_location_accuracy():
    """评估位置识别准确率"""
    # 设置日志
    log_file = setup_logger()
    
    # 加载模型和处理器
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    print(f"加载模型: {model_name}")
    
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="cuda:0"
    )
    
    # 加载测试数据
    test_file = "raw_data_collection/data/qwen_format_v3/test.jsonl"
    print(f"加载测试数据: {test_file}")
    data = load_jsonl(test_file)
    
    # 统计结果
    total = len(data)
    correct = 0
    wrong = 0
    
    # 记录所有案例
    all_cases = []
    
    # 批量处理
    batch_size = 4
    for i in tqdm(range(0, len(data), batch_size), desc="评估进度"):
        batch_data = data[i:i + batch_size]
        batch_messages = []
        batch_image_paths = []
        batch_ground_truths = []
        batch_items = []
        
        # 准备批次数据
        for item in batch_data:
            # 提取用户消息
            try:
                messages = item['messages']
                system_message = messages[0]
                user_message = messages[1]
                ground_truth = item['answer']
                
                # 提取图像路径和真实答案
                image_path = None
                for content in user_message['content']:
                    if content['type'] == 'image':
                        image_path = content['image']
                
                # 提取真实答案
                
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
                prompt_messages = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": system_message["content"][0]["text"]}
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": user_message["content"][0]["text"]}
                        ]
                    }
                ]
                
                batch_messages.append(prompt_messages)
                batch_image_paths.append(image_path)
                batch_ground_truths.append(ground_truth)
                batch_items.append(item)
            
            except Exception as e:
                print(f"处理数据项时出错: {str(e)}")
                continue
        
        if not batch_messages:
            continue
        
        # 批量生成回答
        try:
            # 准备输入
            texts = [processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ) for messages in batch_messages]
            
            # 进行推理
            inputs = processor(
                text=texts,
                images=[messages[1]['content'][0]['image'] for messages in batch_messages],
                return_tensors="pt",
                padding=True,
            )
            
            
            # 生成回答
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=1024)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]
            
            responses = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            # 处理每个回答
            for response, image_path, ground_truth, item in zip(responses, batch_image_paths, batch_ground_truths, batch_items):
                # 构建用于验证的完整响应
                verification_content = "<answer>" + response + "</answer>"
                
                # 使用location_verifier中的验证函数
                score = verify_location(verification_content, ground_truth)
                
                
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
    
    # 计算准确率
    accuracy = correct / total if total > 0 else 0
    
    # 打印统计结果
    print(f"\n总样本数: {total}")
    print(f"正确样本数: {correct} ({accuracy:.2%})")
    print(f"错误样本数: {wrong} ({(1-accuracy):.2%})")
    
    # 保存所有案例
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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

    return accuracy

if __name__ == "__main__":
    print("\n开始位置识别准确率评估...")
    accuracy = evaluate_location_accuracy()
    print(f"最终准确率: {accuracy:.2%}") 