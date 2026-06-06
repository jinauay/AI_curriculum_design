import re

def clean_text(text):
    """
    清理推文文本
    """
    text = re.sub(r"http\S+", "", text)  # 去掉 URL
    text = re.sub(r"@\w+", "", text)     # 去掉 @用户
    text = re.sub(r"#\w+", "", text)     # 去掉话题
    text = re.sub(r"\s+", " ", text)     # 多空格替换为1个
    return text.strip()