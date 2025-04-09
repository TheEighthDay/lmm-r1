import json
import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import time
import re
import sys
import argparse
from PIL import Image
from datetime import datetime
import requests
import base64
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpt4o_mini import make_request
import genai
from io import BytesIO

# 条件导入，根据选择的推理引擎
try:
    from vllm import LLM, SamplingParams
    vllm_available = True
except ImportError:
    vllm_available = False

COT_GROLOC = '''
Question1: Are the reprominent natural features, such as specific types of vegetation, landforms (e.g., mountains, hills, plains), or soil characteristics, that provide clues about the geographical region? 
Question2: Are there any culturally, historically, or architecturally significant landmarks, buildings, or structures, or are there any inscriptions or signs in a specific language or script that could help determine the country or region? 
Question3: Are there distinctive road-related features, such as traffic direction (e.g., left-hand or right hand driving), specific types of bollards, unique utility pole designs, or license plate colors and styles, which countries are known to have these characteristics? 
Question4: Are there observable urban or rural markers (e.g., street signs, fire hydrants guideposts) , or other infrastructure elements, that can provide more specific information about the country or city? 
Question5: Are there identifiable patterns in sidewalks (e.g., tile shapes, colors, or arrangements), clothing styles worn by people, or other culturally specific details that can help narrow down the city or area?

Let's think step by step. Based on the question I provided, locate the location of the picture as accurately as possible. For example: the presence of tropical rainforests, palm trees, and red soil indicates a tropical climate... Signs in Thai, right-side traffic, and traditional Thai architecture further suggest it is in Thailand... Combining these clues, this image was likely taken in a city in Bangkok, Thailand, Asia.
Therefore, the answer is:
<answer>$thailand,bangkok$</answer>
'''

COT_OLD = '''
Let's think step by step. Locate the location of the picture as accurately as possible. For example: the presence of tropical rainforests, palm trees, and red soil indicates a tropical climate... Signs in Thai, right-side traffic, and traditional Thai architecture further suggest it is in Thailand... Combining these clues, this image was likely taken in a city in Bangkok, Thailand, Asia.
Therefore, the answer is:
<answer>$thailand,bangkok$</answer>
'''

COT = '''
Let's think step by step. Locate the location of the picture as accurately as possible. 
Please perform forensic-level detail analysis on the image: First, systematically scan all visual elements including but not limited to architectural styles (roof shapes, materials, color schemes), vegetation types (tree species morphology, leaf characteristics), climate indicators (light angle, cloud formations, precipitation traces), traffic signs (road sign languages, symbol standards, vehicle models), textual information (signage fonts, language characters, regional abbreviations), and geographical features (mountain ranges, water body formations, soil coloration). Then construct an element correlation network: e.g., matching vegetation distribution with climate zones, aligning architectural designs with local cultural patterns, and comparing traffic sign standards against international norms. Pay special attention to microscopic clues: insect species, lichen growth orientation, satellite dish elevation angles, power grid voltage markings. After cross-verification to eliminate contradictory information, compare unique feature combinations against geographical databases (e.g., specific tree species + red soil + left-hand drive vehicles + Gothic spire architecture). Finally, output conclusions based on highest-probability matches that satisfy dual verification of both physical geography and human cultural elements. 
Finally, a guess answer must be given as: <answer>$South Africa, Western Cape Province$</answer>
'''



def setup_logger(output_prefix=None):
    """设置日志记录器"""
    # 创建logs目录
    os.makedirs('logs', exist_ok=True)
    
    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_prefix:
        log_file = f'logs/{output_prefix}_analysis.log'
    else:
        log_file = f'logs/analysis_{timestamp}.log'
    
    # 创建日志文件
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"分析开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
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

def verify_location_with_gpt4o(pred_answer, true_answer):
    """验证位置信息是否匹配
    
    Args:
        pred_answer: 预测答案，可能包含标签 <answer>...</answer>
        true_answer: 真实答案
        
    Returns:
        float: 1.0 如果匹配，0.0 如果不匹配
    """
    # 尝试提取 <answer> 标签中的内容
    try:
        answer_match = re.search(r"<answer>\$(.*?)\$</answer>", pred_answer)
        if answer_match:
            pred_answer = answer_match.group(1).strip()
        else:
            # 尝试其他可能的格式
            answer_match = re.search(r"<answer>(.*?)</answer>", pred_answer)
            if answer_match:
                pred_answer = answer_match.group(1).strip()
            # 如果没有匹配到标签，使用原始文本
    except Exception as e:
        print(f"提取答案时出错: {str(e)}")
        # 使用原始文本，不进行提取
    
    # 如果不在列表中，使用 GPT4O 进行验证
    prompt = f"""请判断以下两个位置是否指的是同一个地方，要求1. 国家（country）2. 行政省份/州（administrative_area_level_1），完全一致才算同一个地方，真实位置中如果administrative_area_level_1存在多个别名，用/分隔，只要有一个正确就行：
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

def load_jsonl(file_path):
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def generate_with_gemini(image_path, api_key, cot=None, model_name=None, max_retries=3, retry_delay=2):
    """使用Gemini API生成预测地址
    
    Args:
        image_path: 图片路径
        api_key: Gemini API密钥
        cot: 是否使用推理提示
        max_retries: 最大重试次数
        retry_delay: 重试延迟时间（秒）
        
    Returns:
        str: 生成的预测地址
    """
    # 配置API
    genai.configure(api_key=api_key)
    
    for attempt in range(max_retries):
        try:
            # 加载图片
            try:
                if image_path.startswith(('http://', 'https://')):
                    response = requests.get(image_path)
                    image = Image.open(BytesIO(response.content))
                else:
                    image = Image.open(image_path)
            except Exception as e:
                print(f"加载图片失败: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return None
            
            # 构建提示
            if not cot:
                prompt = """Please analyze this picture: 
1. The country where this picture was taken (country)
2. The province/state where this picture was taken (administrative_area_level_1)
3. Country and administrative area level 1 (in English).

Please answer in the format of <answer>$country,administrative_area_level_1$</answer>. 
Even if you cannot analyze, give a clear answer including country and administrative area level 1."""
            else:
                prompt = COT + """Please analyze this picture: 
1. The country where this picture was taken (country)
2. The province/state where this picture was taken (administrative_area_level_1)
3. Country and administrative area level 1 (in English).

Please answer in the format of <answer>$country,administrative_area_level_1$</answer>. 
Even if you cannot analyze, give a clear answer including country and administrative area level 1."""
            
            # 创建模型实例
            model = genai.GenerativeModel(model_name)
            
            # 生成响应
            response = model.generate_content([prompt, image])
            
            # 检查响应
            if response and hasattr(response, 'text'):
                return response.text
            else:
                raise Exception("Empty or invalid response")
                
        except Exception as e:
            print(f"Gemini API请求失败: {str(e)}")
            if attempt < max_retries - 1:
                print(f"等待{retry_delay}秒后重试...")
                time.sleep(retry_delay)
                continue
            return None
    
    return None




def generate_with_doubao(image_path, api_key, cot=None, doubao_model_name=None, max_retries=3, retry_delay=2):
    """使用豆包API生成预测地址
    
    Args:
        image_path: 图片路径
        api_key: 豆包API密钥
        cot: 是否使用推理提示
        max_retries: 最大重试次数
        retry_delay: 重试延迟时间（秒）
        
    Returns:
        str: 生成的预测地址
    """
    for attempt in range(max_retries):
        try:
            # 读取图片并转换为base64
            with open(image_path, 'rb') as f:
                encoded_image = base64.b64encode(f.read())
            encoded_image_text = encoded_image.decode('utf-8')
            base64_image = f"data:image/jpeg;base64,{encoded_image_text}"
            
            # 构建消息
            if not cot:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "\n Please analyze this picture: 1. The country where this picture was taken (country) 2. The province/state where this picture was taken (administrative_area_level_1) 3. Country and administrative area level 1 (in English). Please answer in the format of <answer>$country,administrative_area_level_1$</answer>. Even if you cannot analyze, give a clear answer including country and administrative area level 1. \n "
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": base64_image
                                }
                            }
                        ]
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "\n Please analyze this picture: 1. The country where this picture was taken (country) 2. The province/state where this picture was taken (administrative_area_level_1) 3. Country and administrative area level 1 (in English). Please answer in the format of <answer>$country,administrative_area_level_1$</answer>. Even if you cannot analyze, give a clear answer including country and administrative area level 1. \n " + COT
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": base64_image
                                }
                            }
                        ]
                    }
                ]
            
            # 发送请求
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
                headers=headers,
                json={
                    "model": doubao_model_name,
                    "messages": messages
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                print(result)
                if 'choices' in result and len(result['choices']) > 0:
                    response_text = result['choices'][0]['message']['content']
                    print(response_text)
                    return response_text
            elif response.status_code == 429:  # Rate limit
                print(f"请求频率限制，等待{retry_delay}秒后重试...")
                time.sleep(retry_delay)
                continue
            else:
                print(f"豆包API请求失败！状态码: {response.status_code}")
                print(f"错误信息: {response.text}")
                if attempt < max_retries - 1:
                    print(f"第{attempt + 1}次重试...")
                    time.sleep(retry_delay)
                    continue
                return None
                
        except Exception as e:
            print(f"豆包API请求出错: {str(e)}")
            if attempt < max_retries - 1:
                print(f"第{attempt + 1}次重试...")
                time.sleep(retry_delay)
                continue
            return None
            
    return None

def analyze_model(model_name="Qwen/Qwen2.5-VL-7B-Instruct", 
                 test_file="/data/phd/tiankaibin/dataset/data/test.jsonl",
                 batch_size=8,
                 output_prefix=None,
                 output_file=None,
                 cot=None,
                 inference_engine="vllm",
                 pipeline_parallel=False,
                 doubao_api_key=None,
                 use_default_system=False,
                 doubao_model_name=None):
    """分析模型效果"""
    # 设置日志
    log_file = setup_logger(output_prefix)
    
    # 加载模型和处理器
    print(f"加载模型: {model_name}")
    print(f"推理引擎: {inference_engine}")
    print(f"Pipeline并行: {'是' if pipeline_parallel else '否'}")
    
    if inference_engine == "doubao":
        if not doubao_api_key:
            raise ValueError("使用豆包API需要提供API密钥")
        print("使用豆包API进行推理")
    else:
        processor = AutoProcessor.from_pretrained(model_name, padding_side='left')
        
        if inference_engine == "vllm":
            if not vllm_available:
                raise ImportError("vLLM 库不可用，请安装 vllm 或选择 transformers 引擎")
            
            # 使用vLLM加载模型
            if pipeline_parallel:
                # 设置4卡pipeline并行
                llm = LLM(
                    model=model_name,
                    limit_mm_per_prompt={"image": 10, "video": 10},
                    dtype="auto",
                    gpu_memory_utilization=0.8,
                    pipeline_parallel_size=4,  # pipeline并行大小
                    trust_remote_code=True,
                )
            else:
                llm = LLM(
                    model=model_name,
                    limit_mm_per_prompt={"image": 10, "video": 10},
                    dtype="auto",
                    gpu_memory_utilization=0.8,
                    trust_remote_code=True,
                )
            
            # 设置采样参数
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.8,
                repetition_penalty=1.05,
                max_tokens=2048,
                stop_token_ids=[],
            )
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype="auto", device_map="cuda:0"
            )
    
    # 加载测试数据
    print(f"加载测试数据: {test_file}")
    data = load_jsonl(test_file)
    
    # 统计结果
    total = len(data)
    correct = 0
    wrong = 0
    
    # 记录所有案例
    all_cases = []
    
    # 批量处理
    print(f"使用批处理大小: {batch_size}")
    for i in tqdm(range(0, len(data), batch_size), desc="评估进度"):
        batch_data = data[i:i + batch_size]
        batch_messages = []
        batch_image_paths = []
        batch_items = []
        
        if inference_engine == "vllm":
            batch_inputs = []
        
        # 准备批次数据
        responses = []
        for item in batch_data:
            # 解析消息
            message = json.loads(item['message'])
            image_path = message[1]['content'][0]['image']
            
            if not os.path.exists(image_path):
                print(f"图像文件不存在: {image_path}")
                wrong += 1
                all_cases.append({
                    'image': image_path,
                    'prediction': '图像文件不存在',
                    'ground_truth': item['answer'],
                    'is_correct': False,
                    'error_type': 'file_not_found'
                })
                # 记录结果
                log_result(log_file, image_path, '图像文件不存在', item['answer'], 0.0, item, 'file_not_found')
                continue
                
            try:
                image = Image.open(image_path)
            except Exception as e:
                print(f"加载图像失败: {image_path}, 错误: {str(e)}")
                wrong += 1
                all_cases.append({
                    'image': image_path,
                    'prediction': f'加载图像失败: {str(e)}',
                    'ground_truth': item['answer'],
                    'is_correct': False,
                    'error_type': 'image_load_error'
                })
                # 记录结果
                log_result(log_file, image_path, f'加载图像失败: {str(e)}', item['answer'], 0.0, item, 'image_load_error')
                continue
            
            if inference_engine == "doubao":
                # 使用豆包API生成预测
                if "doubao" in doubao_model_name:
                    response = generate_with_doubao(image_path, doubao_api_key, cot, doubao_model_name)
                else:
                    response = generate_with_gemini(image_path, doubao_api_key, cot, doubao_model_name)
                responses.append(response)
                batch_image_paths.append(image_path)
                batch_items.append(item)

            else:
                # 构建消息
                if not cot:
                    if not use_default_system:
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                {
                                    "type": "image",
                                    "image": image,
                                },
                                {"type": "text", "text": "Please analyze this picture: 1. The country where this picture was taken (country) 2. The province/state where this picture was taken (administrative_area_level_1) 3. Country and administrative area level 1 (in English). Please answer in the format of <answer>$country,administrative_area_level_1$</answer>. Even if you cannot analyze, give a clear answer including country and administrative area level 1. \n"}
                            ],
                            }
                        ]
                    else:
                        messages = [
                            {
                                "role": "system",
                                "content": "You are a helpful assistant good at solving problems with step-by-step reasoning. You should first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags."
                            },
                            {
                                "role": "user",
                                "content": [
                                {
                                    "type": "image",
                                    "image": image,
                                },
                                {
                                    "type": "text", "text": "Please analyze this picture: 1. The country where this picture was taken (country) 2. The province/state where this picture was taken (administrative_area_level_1) 3. Country and administrative area level 1 (in English). Please answer in the format of <answer>$country,administrative_area_level_1$</answer>. Even if you cannot analyze, give a clear answer including country and administrative area level 1. \n"
                                }
                            ],
                            }
                        ]

                else:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": "\n Please analyze this picture: 1. The country where this picture was taken (country) 2. The province/state where this picture was taken (administrative_area_level_1) 3. Country and administrative area level 1 (in English). Please answer in the format of <answer>$country,administrative_area_level_1$</answer>. Must Guess a clear answer including country and administrative area level 1. \n " + COT},
                            ],
                        }
                    ]

                if inference_engine == "vllm":
                    # 处理消息为vLLM格式
                    prompt = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    
                    # 处理图像数据
                    image_inputs, video_inputs = process_vision_info(messages)
                    
                    mm_data = {}
                    if image_inputs is not None:
                        mm_data["image"] = image_inputs
                    
                    # 构建vLLM输入
                    llm_input = {
                        "prompt": prompt,
                        "multi_modal_data": mm_data,
                    }
                    batch_inputs.append(llm_input)
                
                batch_messages.append(messages)
                batch_image_paths.append(image_path)
                batch_items.append(item)
        
        if not batch_messages and inference_engine != "doubao":
            continue
            
        # 批量生成回答
        try:
            
            
            if inference_engine == "vllm":

                outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
                responses = [output.outputs[0].text for output in outputs]
                    
            elif inference_engine == "transformers":
                # 准备输入
                texts = [processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                ) for messages in batch_messages]
                
                image_inputs, video_inputs = process_vision_info(batch_messages)
                inputs = processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                
                # 确保所有输入都在 cuda:0 上
                inputs = {k: v.to("cuda:0") if hasattr(v, 'to') else v for k, v in inputs.items()}
                
                # 生成回答
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs, 
                        max_new_tokens=2048
                    )
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
                ]
                responses = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            
            # 处理每个回答

            for response, image_path, item in zip(responses, batch_image_paths, batch_items):
                print("__________________________")
                print(response)
                print("__________________________")

                pred_answer = response                
                # 使用 GPT4O 验证位置是否匹配
                score = verify_location_with_gpt4o(pred_answer, item['answer'])
                
                # 记录结果
                error_type = None if score == 1.0 else 'location_mismatch'
                log_result(log_file, image_path, response, item['answer'], score, item, error_type)
                
                # 统计结果
                if score == 1.0:
                    correct += 1
                    all_cases.append({
                        'image': image_path,
                        'prediction': response,
                        'ground_truth': item['answer'],
                        'is_correct': True,
                        'error_type': None
                    })
                else:
                    wrong += 1
                    all_cases.append({
                        'image': image_path,
                        'prediction': response,
                        'ground_truth': item['answer'],
                        'is_correct': False,
                        'error_type': 'location_mismatch'
                    })
                
        except Exception as e:
            print(f"模型推理失败: {str(e)}")
            for image_path, item in zip(batch_image_paths, batch_items):
                wrong += 1
                all_cases.append({
                    'image': image_path,
                    'prediction': f'模型推理失败: {str(e)}',
                    'ground_truth': item['answer'],
                    'is_correct': False,
                    'error_type': 'model_inference_error'
                })
                # 记录结果
                log_result(log_file, image_path, f'模型推理失败: {str(e)}', item['answer'], 0.0, item, 'model_inference_error')
        
        # 清理GPU缓存，避免内存泄漏
        torch.cuda.empty_cache()
    
    # 计算准确率
    accuracy = correct / total if total > 0 else 0
    
    # 打印统计结果
    print(f"\n总样本数: {total}")
    print(f"正确样本数: {correct} ({accuracy:.2%})")
    print(f"错误样本数: {wrong} ({(1-accuracy):.2%})")
    
    # 保存所有案例
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cases_file = f'logs/all_cases_{timestamp}.json'
    if output_prefix:
        cases_file = f'logs/{output_prefix}_cases.json'
    
    with open(cases_file, 'w', encoding='utf-8') as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)
    print(f"\n所有案例已保存到: {cases_file}")
    
    # 记录分析结束时间
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n分析结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总样本数: {total}\n")
        f.write(f"正确数: {correct} ({correct/total*100:.2f}%)\n")
        f.write(f"错误数: {wrong} ({wrong/total*100:.2f}%)\n")

    # 如果指定了输出文件，保存结果
    if output_file:
        output_data = {
            "accuracy": float(accuracy),
            "total_samples": total,
            "correct_samples": correct,
            "wrong_samples": wrong,
            "model_name": model_name,
            "inference_engine": inference_engine,
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
    parser.add_argument('--test_file', type=str, default="/data/phd/tiankaibin/dataset/data/test.jsonl",
                        help='测试数据文件路径')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批处理大小，较大的值可以加速推理')
    parser.add_argument('--output_prefix', type=str, default=None,
                        help='输出文件前缀')
    parser.add_argument('--output_file', type=str, default=None,
                        help='结果文件路径，用于保存准确率和预测结果')
    parser.add_argument('--cot', action='store_true',
                        help='是否使用推理提示')
    parser.add_argument('--inference_engine', type=str, default="vllm", choices=["vllm", "transformers", "doubao"],
                        help='推理引擎: vllm, transformers 或 doubao')
    parser.add_argument('--pipeline_parallel', action='store_true',
                        help='是否使用4卡pipeline并行')
    parser.add_argument('--doubao_api_key', type=str, default=None,
                        help='豆包API密钥，使用豆包API时需要提供')
    parser.add_argument('--doubao_model_name', type=str, default="doubao-1.5-vision-pro-32k-250115",
                        help='豆包模型名称')
    parser.add_argument('--use_default_system', action='store_true',
                        help='使用训练过的use_default_system')
    
    args = parser.parse_args()
    
    print("\n开始运行完整分析...")
    print(f"模型: {args.model_name}")
    print(f"批处理大小: {args.batch_size}")
    print(f"推理引擎: {args.inference_engine}")
    print(f"Pipeline并行: {'是' if args.pipeline_parallel else '否'}")
    
    # 设置GPU内存限制
    if torch.cuda.is_available():
        # 清理缓存
        torch.cuda.empty_cache()
        # 设置内存使用率为0.8
        torch.cuda.set_per_process_memory_fraction(0.95)
    
    analyze_model(
        model_name=args.model_name,
        test_file=args.test_file,
        batch_size=args.batch_size,
        output_prefix=args.output_prefix,
        output_file=args.output_file,
        cot=COT if args.cot else None,
        inference_engine=args.inference_engine,
        pipeline_parallel=args.pipeline_parallel,
        doubao_api_key=args.doubao_api_key,
        use_default_system=args.use_default_system,
        doubao_model_name=args.doubao_model_name
    )

# 使用示例:
# CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_zero_7b_acc.py --batch_size 4 --output_file results/qwen7b_eval_results.json --inference_engine vllm --pipeline_parallel 
# CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_zero_7b_acc.py --batch_size 4 --output_file results/qwen7b_eval_results_cot.json --cot --inference_engine vllm --pipeline_parallel

# CUDA_VISIBLE_DEVICES=4,5,6,7  python eval_zero_7b_acc.py --batch_size 4 --output_file results/qwen32b_eval_results.json --inference_engine vllm --model_name=Qwen/Qwen2.5-VL-32B-Instruct --pipeline_parallel

# CUDA_VISIBLE_DEVICES=0 python eval_zero_7b_acc.py --batch_size 4 --output_file results/multiver_RL_eval_results.json --inference_engine transformers  --model_name=/data/phd/tiankaibin/experiments_multi_system/checkpoints/lmm-r1-multi-system/ckpt/global_step150_hf --use_default_system

# CUDA_VISIBLE_DEVICES=3 python eval_zero_7b_acc.py --batch_size 4 --output_file results/deepscaler_RL_eval_results.json --inference_engine vllm  --model_name=/data/phd/tiankaibin/lmm-r1/experiments_deepscaler/checkpoints/lmm-r1-deepscaler/ckpt/global_step200_hf

# CUDA_VISIBLE_DEVICES=1 python eval_zero_7b_acc.py --batch_size 4 --output_file results/doubao_eval_results_cot.json --inference_engine doubao --doubao_api_key=xx --cot
# CUDA_VISIBLE_DEVICES=0 python eval_zero_7b_acc.py --batch_size 4 --output_file results/qwen7b_eval_results_cot_tkbnew.json --inference_engine vllm --model_name=Qwen/Qwen2.5-VL-7B-Instruct --cot
