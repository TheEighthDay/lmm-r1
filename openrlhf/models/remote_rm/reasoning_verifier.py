import json
from typing import List, Dict
from vllm import LLM, SamplingParams
from loguru import logger
from flask import Flask, jsonify, request
import random
from argparse import ArgumentParser
import re
from concurrent import futures
import Levenshtein

app = Flask(__name__)

# 初始化Qwen模型
model = LLM(
    model="Qwen/Qwen2.5-3B-Instruct",  # 使用量化版本的Qwen模型
    trust_remote_code=True,
    dtype="auto",
    gpu_memory_utilization=0.9,
)

# 初始化线程池
reasoning_verify_executor = futures.ThreadPoolExecutor(max_workers=32)

# 格式验证的正则表达式
format_pattern = r"^<think>(?:(?!</think>).)*</think>\s*<answer>(?:(?!</answer>).)*</answer>\Z"

def verify_format(content):
    """
    Verify if the string meets the format requirements:
    - Must start with <think> and end with </answer>
    - Must contain exactly one pair of <think>...</think> and <answer>...</answer> tags
    - No extra characters allowed between </think> and <answer> tags
    """
    think_count = content.count("<think>")
    answer_count = content.count("<answer>")
    comma_count = content.count(",")
    return bool(re.match(format_pattern, content, re.DOTALL)) and think_count == 1 and answer_count == 1 and comma_count == 1

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
    使用 Levenshtein 距离，如果两个单词的距离小于3，则认为匹配
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
        sol: 真实答案，格式为 $country,region1/region2/region3$
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

def extract_reasoning(content: str) -> str:
    """从内容中提取推理过程"""
    try:
        start = content.find("<think>") + len("<think>")
        end = content.find("</think>")
        return content[start:end].strip()
    except:
        return ""

def verify_reasoning(reasoning: str, answer: str) -> float:
    """使用Qwen模型验证推理过程是否合理"""
    prompt = f"""请仔细分析以下推理过程是否合理，并给出评分。
推理过程：{reasoning}
最终答案：{answer}

请用方括号[]输出结果，例如：
[1] 推理过程合理
或
[0] 推理过程不合理"""

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.1,
        max_tokens=10,
    )

    logger.info(f"开始验证推理过程:\n推理过程: {reasoning}\n最终答案: {answer}")
    
    outputs = model.generate(prompt, sampling_params)
    result = outputs[0].outputs[0].text.strip()
    
    logger.info(f"Qwen模型输出: {result}")
    
    # 使用正则表达式提取方括号中的数字
    match = re.search(r'\[(\d+)\]', result)
    if match:
        try:
            reward = float(match.group(1))
            logger.info(f"成功解析结果: {reward}")
            return reward
        except:
            logger.warning(f"解析结果失败: {result}")
            return 0.0
    else:
        logger.warning(f"未找到有效结果: {result}")
        return 0.0

@app.route("/get_reward", methods=["POST"])
def get_reward():
    data = request.get_json()
    if "query" not in data:
        return jsonify({"error": "queries field is required"}), 400
    
    reasoning_rewards = []
    format_rewards = []
    location_rewards = []
    reasoning_rewards_futures = []
    location_rewards_futures = []
    
    for q, problem in zip(data["query"], data["prompts"]):
        if problem is None:
            return jsonify({"error": f"problem not found from {q}"}), 400
            
        # 格式验证
        format_reward = float(verify_format(q)) * 0.5
        format_rewards.append(format_reward)
        
        # 位置验证
        answer = problem_to_answer.get(problem, "")
        if not answer:
            location_rewards.append(0.0)
            reasoning_rewards.append(0.0)
            continue
            
        # 使用线程池异步处理位置验证
        location_future = reasoning_verify_executor.submit(verify_location, q, answer)
        location_rewards_futures.append(location_future)
        
        # 推理验证
        reasoning = extract_reasoning(q)
        if not reasoning:
            reasoning_rewards.append(0.0)
            continue
            
        # 使用线程池异步处理推理验证
        reward_future = reasoning_verify_executor.submit(verify_reasoning, reasoning, answer)
        reasoning_rewards_futures.append(reward_future)
        
        # 随机打印一些样本用于调试
        if random.randint(1, 20) == 1:
            logger.info(f"Problem: {problem}\nReasoning: {reasoning}\nAnswer: {answer}\nFormat Reward: {format_reward}")
    
    # 收集所有异步结果
    location_rewards = [f.result() for f in location_rewards_futures]
    reasoning_rewards = [f.result() for f in reasoning_rewards_futures]
    
    # 计算最终奖励
    final_rewards = [f + l + r for f, l, r in zip(format_rewards, location_rewards, reasoning_rewards)]
    
    return jsonify({
        "rewards": final_rewards,
        "format_rewards": format_rewards,
        "location_rewards": location_rewards,
        "reasoning_rewards": reasoning_rewards
    })

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default=None, help="Datasets to use (comma separated)", required=True
    )
    parser.add_argument(
        "--input_key", type=str, default="prompt", help="The key name of prompt."
    )
    parser.add_argument("--log_file", type=str, default="reasoning_verifier.log", help="Log file path")
    args = parser.parse_args()
    
    logger.remove()
    logger.add(args.log_file)
    
    # 加载数据集
    problem_to_answer = {}
    for dataset_path in args.dataset.split(','):
        dataset_path = dataset_path.strip()
        if dataset_path.endswith("json"):
            with open(dataset_path, "r") as f:
                dataset = json.load(f)
        elif dataset_path.endswith("jsonl"):
            with open(dataset_path, "r") as f:
                dataset = [json.loads(l) for l in f.readlines()]
        else:
            raise ValueError(f"Unsupported file format for dataset: {dataset_path}")
            
        for item in dataset:
            problem = item[args.input_key]
            answer = item["answer"].strip()
            if answer[0] != "$":
                answer = "$" + answer + "$"
            problem_to_answer[problem] = answer
    
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
    reasoning_verify_executor.shutdown() 