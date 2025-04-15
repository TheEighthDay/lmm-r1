from openai import AzureOpenAI
import time
import base64
from PIL import Image
import requests
from io import BytesIO



def make_request(prompt="",
                 base="xxx",
                 key="xxxx",
                 model="xxx", 
                 max_tokens=2048,
                 system_content=None, 
                 retries=3,
                 image_path=None):
    """调用Azure OpenAI API进行文本或多模态生成
    
    Args:
        prompt: 文本提示
        base: Azure OpenAI API基础URL
        key: API密钥
        model: 模型名称
        max_tokens: 最大生成token数
        system_content: 系统提示
        retries: 最大重试次数
        image_path: 图片路径，可以是本地路径或URL
        
    Returns:
        dict: 包含response和error字段的响应字典
    """
    client = AzureOpenAI(api_version="2024-03-01-preview",
                         api_key=key,
                         azure_endpoint=base)

    response = None
    target_length = int(max_tokens)
    retry_cnt = 0
    backoff_time = 15
    a = time.time()
    
    # 处理图片
    image_content = None
    if image_path:
        try:
            if image_path.startswith(('http://', 'https://')):
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image_path)
            
            # 将图片转换为base64
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            image_content = base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            return {"error": f"图片处理失败: {str(e)}", "response": None}
    
    while retry_cnt <= retries:
        try:
            messages = []
            
            # 添加系统提示
            if system_content is not None:
                messages.append({"role": "system", "content": system_content})
            
            # 构建用户消息
            user_content = []
            if image_content:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_content}"
                    }
                })
            if prompt:
                user_content.append({
                    "type": "text",
                    "text": prompt
                })
            
            messages.append({"role": "user", "content": user_content})
            
            # 调用API
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=target_length,
                temperature=0.7,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None
            )
            break
            
        except Exception as e:
            print(f"API请求失败，尝试 {retry_cnt + 1}/{retries}: {str(e)}")
            time.sleep(backoff_time)
            backoff_time *= 1.5
            retry_cnt += 1

    if response is None or response.choices[0].finish_reason == "length":
        response_text = ""
    else:
        response_text = response.choices[0].message.content

    data = {"response": response_text}
    
    b = time.time()
    print("请求耗时:", b - a)
    print("响应:", data)
    return data