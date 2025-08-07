from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore
from collections import defaultdict

PROMPT_ANS_EXP=('''You are a helpful assistant. You are given a multiple-choice question with optional answers (marked by A, B, C, and so on). Please choose the best answer by responding with its corresponding letter (A, B, C, and so on). 

Question: {question}
Options: {options}

Answer (starting with the corresponding letter (A, B, C, and so on) and then give explanation): ''')

PROMPT_EXP_ANS=('''You are a helpful assistant. You are given a multiple-choice question with optional answers (marked by A, B, C, and so on). Please first give an concise explanation or analysis and then choose the best answer by responding with letter (A, B, C, and so on). 

Question: {question}
Options: {options}

Explanation and Answer: ''')


import re, numpy as np 
def extract_letter(text, verbose_info_level=0, verbose_error_level=0):
    text=text.strip('­--').strip()

    if not text:
        if verbose_info_level:
            print("Input text is empty.")
        return np.nan
    
    patterns = [
        r'^\(?([A-D])[\)\.:]?',
        r'^Option ([A-D])',
        r'^answer: ([A-D])',
        r'\(?([A-D])\)?',
        # r'is \(?([A-D])\)?',
    ]
    
    for pattern in patterns:
        result = re.findall(pattern, text)
        if result:
            if verbose_info_level:
                print(f"Pattern '{pattern}' matched: {result[0]}, in text: {text}")
            return result[0]
    
    res = text.split()[0].rstrip(',')
    if res.endswith('assistant'):
        res = res[:-len('assistant')]
    if res in 'ABCDE':
        if verbose_info_level:
            print(f"Text split result matched: {res}")
        return res
    
    if verbose_error_level:
        print(f"No pattern matched for text: {text}")
    
    return np.nan

import hashlib, os 
def min_unique_length(md5_list):
    md5_list=set(list(md5_list))
    # 从1位字符开始逐步增加截取长度
    for length in range(1, 33):  # MD5 哈希总共32位
        truncated_md5s = [md5[:length] for md5 in md5_list]
        if len(set(truncated_md5s)) == len(md5_list):
            return length
    return 32  # 如果所有字符都一样，返回完整的32位
def calculate_md5(input_data):
    if isinstance(input_data, str) and os.path.isfile(input_data):
        # If input_data is a file path
        hash_md5 = hashlib.md5()
        with open(input_data, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    else:
        # If input_data is a string
        return hashlib.md5(input_data.encode()).hexdigest()
def unique_by_md5(files):
    """检查文件的MD5值并标记重复文件"""
    md5_dict = {}
    duplicates = []

    for file_path in files:
        if os.path.exists(file_path):
            md5_value = calculate_md5(file_path)
            if md5_value in md5_dict:
                duplicates.append(file_path)
            else:
                md5_dict[md5_value] = file_path
        else:
            print(f"文件不存在: {file_path}")
    files = list(md5_dict.values())
    return sorted(files, key=os.path.getctime, reverse=True)
     

def my_prt(*message):
    import inspect
    import torch.distributed as dist, random, numpy as np 
    if dist.is_initialized() and dist.get_rank() == 0 or not dist.is_initialized():
        caller_frame = inspect.currentframe().f_back
        file_name = caller_frame.f_code.co_filename
        line_number = caller_frame.f_lineno
        print(f"{file_name}:{line_number}", *message)

def clean_nm(s):
    parts_without_extension = s.rsplit('.', 1)[0]
    
    # Split by '/' to get individual directory names
    path_parts = parts_without_extension.split('/')
    
    # List of directory names to exclude
    exclude_dirs = {'home', 'ma-user', 'work', 'xinglu', 'sq-gptq', 'output', 'outputs', 'llm_pretrain'}
    
    # Filter out the excluded directories
    filtered_parts = [part for part in path_parts if part not in exclude_dirs]
    
    # Rejoin the parts to form the final path
    return  '-'.join(filtered_parts).strip('-')

def create_filename():
    from     datetime import datetime
    # 获取当前时间并格式化为字符串，适用于文件名
    return datetime.now().strftime("file_%Y%m%d_%H%M%S.json")

import random, numpy as np, torch 
def set_seed(seed=6):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # following may slow down
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def get_hist(data_path):
    history_data = defaultdict(list)
    ds = datastore.DataStore()
    ds.open_for_scan(data_path)
    while True:  
        # try:
        data = ds.scan_record()
        # except:
        #     continue 
        if not data or (len(data))<2  : 
            break 
        pb = wandb_internal_pb2.Record()
        try:
            pb.ParseFromString(data[1])  
        except  Exception as e: 
            if 'Error parsing message' in str(e):
                continue 
        record_type = pb.WhichOneof("record_type")
        if record_type == "history":
            for item in pb.history.item:
                key = item.key
                value = (item.value_json) 
                history_data[key].append(float(value)) 
                
    return history_data 

import re
import pandas as pd, json 

def get_hist_from_log(data_path):
    if not os.path.exists(data_path): 
        print('not exist ', data_path)
        return None 
    with open(data_path, 'r') as f:
        data_string = f.read()
    # Extract dictionaries from string
    data_list = re.findall(r'\{[^}]+\}', data_string)
    data_dicts = []
    for item in data_list:
        try:
            item = item.replace("'", '"')
            data_dicts.append(json.loads(item))
        except json.JSONDecodeError as e:
            print(f' {data_path} : Error parsing line: {item} in\nException: {e}')
            continue

    # Create DataFrame
    df = pd.DataFrame(data_dicts)
    return df 

def get_next_folder_name(data_path, target_folder):
    path_parts = data_path.strip('/').split('/')
    if target_folder in path_parts:
        target_index = path_parts.index(target_folder)
        if target_index + 1 < len(path_parts):
            return path_parts[target_index + 1]
    return None

