import os
import google.generativeai as genai
from PIL import Image
import time
import requests
from io import BytesIO

#'gemini-2.0-pro-exp-02-05', 'gemini-2.0-flash-thinking-exp-01-21'

def make_request(prompt=None, image_path=None, api_key=None, model_name="gemini-pro-vision", max_retries=3, retry_delay=2):
    """
    调用 Gemini API 进行文本或多模态生成
    
    Args:
        prompt (str): 文本提示
        image_path (str): 图片路径，可以是本地路径或URL
        api_key (str): Gemini API密钥，如果为None则从环境变量获取
        model_name (str): 模型名称，默认为"gemini-pro-vision"
        max_retries (int): 最大重试次数
        retry_delay (int): 重试延迟时间（秒）
        
    Returns:
        dict: 包含response和error字段的响应字典
    """
    # 获取API密钥
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"error": "API key not found", "response": None}
    
    # 配置API
    genai.configure(api_key=api_key)
    
    for attempt in range(max_retries):
        try:
            # 选择模型
            if image_path:
                model = genai.GenerativeModel(model_name)
                
                # 加载图片
                try:
                    if image_path.startswith(('http://', 'https://')):
                        response = requests.get(image_path)
                        image = Image.open(BytesIO(response.content))
                    else:
                        image = Image.open(image_path)
                except Exception as e:
                    return {"error": f"Failed to load image: {str(e)}", "response": None}
                
                # 生成响应
                response = model.generate_content([prompt, image] if prompt else [image])
            else:
                # 纯文本请求使用gemini-pro
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
            
            # 检查响应
            if response and hasattr(response, 'text'):
                return {"response": response.text, "error": None}
            else:
                raise Exception("Empty or invalid response")
                
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"请求失败，{retry_delay}秒后重试: {str(e)}")
                time.sleep(retry_delay)
                continue
            return {"error": f"API request failed: {str(e)}", "response": None}
    
    return {"error": "Max retries exceeded", "response": None}

# 使用示例
if __name__ == "__main__":
    # 文本示例
    text_result = make_request(prompt="What is the capital of France?")
    print("Text Response:", text_result)
    
    # 图片示例
    image_result = make_request(
        prompt="Describe this image in detail.",
        image_path="path/to/your/image.jpg"
    )
    print("Image Response:", image_result)

# {"message": "[{\"role\": \"system\", \"content\": \"You are a helpful assistant good at solving problems with step-by-step reasoning. You should first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags.\"}, {\"role\": \"user\", \"content\": [{\"type\": \"image\", \"image\": \"/data/phd/tiankaibin/dataset/data/streetview_images_first_tier_cities/testaccio_rome_italy_h45_r100_20250317_183133.jpg\"}, {\"type\": \"text\", \"text\": \"In which country and within which first-level administrative region of that country was this picture taken?Please answer in the format of <answer>$country,administrative_area_level_1$</answer>?\"}]}]", "answer": "$italy,lazio/lazio/latium/latium$"}