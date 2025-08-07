from utils.otherutils import my_prt
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Union
import os,glob
from utils.otherutils import calculate_md5
from utils.otherutils import PROMPT_ANS_EXP, min_unique_length
import re
from utils import otherutils
import pandas as pd
import itertools
from transformers import AutoTokenizer
from copy import deepcopy 
RENAME_MAP={
            "question":"Question",
            "options":"Options",
            "choices":"Options",                
            "evidence":"Evidence",
            "answer":"Gold",
            }
def split_text_into_chunks(text, tokenizer,  max_tokens=3000, shift=0):
    # 对整个文本进行tokenize
    tokens = tokenizer.encode(text)
    # 按照最大tokens数量进行分割
    chunks = []
    for i in range(shift, len(tokens), max_tokens): # TODO about info at boundary??
        chunk_tokens = tokens[i:i + max_tokens]
        if len(chunk_tokens)< max_tokens//6: continue # skip last too short chunk 
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    print("info: seq len ", trainenc.input_ids.shape[1], testenc.input_ids.shape[1])
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        msk = trainenc.attention_mask[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append({
            'input_ids': inp,
            'attention_mask': msk,
            'tar': tar
        })

    random.seed(seed + 1)
    val_loader = []
    for _ in range(nsamples):
        i = random.randint(0, testenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = testenc.input_ids[:, i:j]
        msk = testenc.attention_mask[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        val_loader.append({
            'input_ids': inp,
            'attention_mask': msk,
            'tar': tar
        })

    return trainloader, val_loader

def get_ptb(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation') # test 

    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    val_enc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        msk = trainenc.attention_mask[:, i:j] 
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append(
            {'input_ids':inp, 
             'attention_mask':msk, 
             'tar': tar 
             })

    random.seed(seed+1)
    val_loader = []
    for _ in range(nsamples):
        i = random.randint(0, val_enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = val_enc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        msk = val_enc.attention_mask[:, i:j]
        val_loader.append({
            'input_ids': inp,
            'attention_mask': msk,
            'tar': tar
        })
    
    return trainloader, val_loader

from datasets import load_dataset
import random

def get_c4(nsamples, seed, seqlen, model):
    traindata = load_dataset(
        'allenai/c4', 'en', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    val_data = load_dataset(
        'allenai/c4', 'en', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples): # this is slow
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        assert inp.shape[0]==1
        msk = trainenc.attention_mask[:, i:j] 
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append(
            {'input_ids':inp, 
            #  'text': tokenizer.decode(inp[0]), # this will cause slightly difference if tokneize again, better avoid use it
             'attention_mask':msk, 
             'tar': tar 
             }
        )
        
    random.seed(seed)
    val_loader = []
    for _ in range(nsamples):  # Adjust number of samples if necessary
        while True:
            i = random.randint(0, len(val_data) - 1)
            val_enc = tokenizer(val_data[i]['text'], return_tensors='pt', max_length=seqlen, truncation=True)
            if val_enc.input_ids.shape[1] == seqlen:
                break

        inp = val_enc.input_ids
        msk = val_enc.attention_mask
        tar = inp.clone()
        tar[:, :-1] = -100

        val_loader.append({
            'input_ids': inp,
            'attention_mask': msk,
            'tar': tar
        })
    return trainloader, val_loader 

def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model="facebook/opt-125m"
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    elif 'ptb' in name:
        return get_ptb(nsamples, seed, seqlen, model)
    elif 'c4' in name:
        return get_c4(nsamples, seed, seqlen, model)
    else:
        raise ValueError(name) 
    
def save_tensors_to_hdf5(data_dict, filename, subsets=None):
    """
    Save a dictionary of PyTorch tensors to an HDF5 file.

    Args:
    - data_dict (dict): A dictionary where the keys are strings and the values are PyTorch tensors.
    - filename (str): The filename of the HDF5 file to save the data to.
    """
    import h5py
    # Create a new HDF5 file
    with h5py.File(filename, "w") as f:
        # Create datasets for each tensor in data_dict
        for key, value in data_dict.items():
            if subsets is not None and key not in subsets: continue 
            f.create_dataset(key, data=value)

def load_tensors_from_hdf5(filename):
    import h5py
    import numpy as np
    """
    Load a dictionary of PyTorch tensors from an HDF5 file.

    Args:
    - filename (str): The filename of the HDF5 file to load the data from.

    Returns:
    - data_dict (dict): A dictionary where the keys are strings and the values are PyTorch tensors.
    """
    data_dict = {}
    with h5py.File(filename, "r") as f:
        # Iterate over datasets in the HDF5 file
        for key in f.keys():
            # Load dataset into numpy array
            data = np.array(f[key])
            # Add tensor to data_dict
            data_dict[key] = data
    return data_dict

def load_json(file_path):
    """
    Load JSON data from a file.

    Parameters:
    - file_path (str): The path to the JSON file.

    Returns:
    - dict: The JSON data loaded as a dictionary.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def create_folder_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def dump_json_sync(data, file_path):
    """
    Dump data to a JSON file.

    Parameters:
    - data (dict): The dictionary data to be dumped.
    - file_path (str): The path where the JSON file will be saved.

    """
    my_prt('save to ', file_path)
    with open(file_path, 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def dump_json(data, file_path, async_mode=False):
    import threading 
    if async_mode:
        thread = threading.Thread(target=dump_json_sync, args=(data, file_path))
        thread.start()
        return thread 
    else:
        dump_json_sync(data, file_path)

def load_jsonl(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        for idx, line in enumerate(file):
            try:
                json_obj = json.loads(line.strip())
            except Exception as e:
                my_prt('err',filename, idx+1,  e)
            data.append(json_obj)
    return data

def dump_jsonl(data, filename):
    # 创建文件夹路径
    folder_path = os.path.dirname(filename)
    create_folder_path(folder_path)
    
    # 写入数据到JSONL文件
    with open(filename, 'w', encoding='utf-8') as file:
        for idx, item in enumerate(data):
            try:
                json_str = json.dumps(item, ensure_ascii=False)
                file.write(json_str + '\n')
            except Exception as e:
                my_prt('err', filename, idx + 1, e)

def sort_according_to_latter_list(primary_list, sorting_list):
    # Ensure both lists have the same length
    assert len(primary_list) == len(sorting_list), "Both lists must have the same length"
    
    # Combine both lists into a list of tuples and sort by the second element (sorting_list)
    combined = list(zip(primary_list, sorting_list))
    combined_sorted = sorted(combined, key=lambda x: x[1])
    
    # Unzip the sorted combined list back into two lists
    sorted_primary_list, sorted_sorting_list = zip(*combined_sorted)
    
    return list(sorted_primary_list)

class LongContextDSBase(ABC):
    def __init__(self, *args, **kwargs):
        pass 

    def get_candidates(self) -> List[str]:
        """
        获取所有小说的文件名，去除扩展名，返回列表

        Returns:
            List[str]: 小说文件名列表
        """
        return ['default']

    @abstractmethod
    def get_novel(self) -> str:
        """
        读取指定小说的全文

        Returns:
            str: 小说全文内容
        """
        pass

    @abstractmethod
    def get_qas(self) -> List[Dict[str, Union[str, List[str]]]]:
        """
        加载指定小说的问题和答案数据

        Returns:
            List[Dict[str, Union[str, List[str]]]]: 问题和答案数据列表
            Expect format: list of dict with keys 'Question', 'Answer', 'Options', 'Gold' 
        """
        pass
    
    @staticmethod
    def clean_up_choices(choices):
        cleaned_choices = []
        if isinstance(choices, dict):
            choices = list(choices.values())
        for ch in choices:
            ch = ch.strip()
            ch = re.sub(r'^[A-D]\)\s*', '', ch).strip() # remove A)
            ch = re.sub(r'^[A-D]\.\s*', '', ch).strip() # remove A.
            ch = re.sub(r'^\([A-D]\)\.?\s*', '', ch).strip() # remove (A) and (A).
            # cannot remove "A blabla"
            cleaned_choices.append(ch)
        return cleaned_choices
    
    @staticmethod
    def convert_choices_to_str(choices, pos="ABCD"):
        res_str = ""
        idxs = "ABCDEFGHI"
        assert len(choices) <= len(idxs)
        choices=deepcopy(choices)
        choices = LongContextDSBase.clean_up_choices(choices)
        choices = sort_according_to_latter_list(choices, pos)

        for ch, idx in zip(choices, idxs):
            assert not ch.startswith(f'{idx}. ') and not ch.startswith(f"{idx})")
            res_str += f"({idx}) " + ch + "\n"
        return res_str.strip("\n")

    def get_prompts(self) -> List[str]:
        """
        生成指定小说的问题提示

        Returns:
            List[str]: 问题提示列表
        """
        prompts=[]
        df=self.dfs[self.option]
        for q,chs in zip(df['Question'], df['Options']):
            prompts.append(
                PROMPT_ANS_EXP.format(question=q, options = self.convert_choices_to_str(chs))
            )
        return prompts 
    
    def get_permed_qas(self):
        res=[]
        for now in self.get_qas():
            num_op = len(now['Options'])
            if num_op==1: continue 
            ABCD="ABCDEFG"[:num_op]
            if now["Gold"] not in ABCD: continue 
            for pos in (itertools.permutations(ABCD)): 
                mapping = dict(zip(ABCD, pos)) 
                tmp=deepcopy(now)
                tmp["Gold"]=mapping[tmp["Gold"]] 
                tmp['pos']=pos
                
                tmp["prompt"] = PROMPT_ANS_EXP.format(question=tmp['Question'],
                                                      options=self.convert_choices_to_str(tmp['Options'], pos)  ) 
                res.append(tmp) 
        return res


class BasicQA(LongContextDSBase):
    prompts = [
        'Hi, who are u?',
        'What is the primary function of the hippocampus in the human brain?\nA) To regulate blood pressure\nB) To control voluntary movement\nC) To process and retrieve memories\nD) To digest proteins',
     
    ]
    prompts_template="你是一个助手，请遵循用户的指令。"
    
    def __init__(self, tker=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tker=tker
        if tker is None:
            self.set_tker()
    
    def set_tker(self, model_path='/home/ma-user/work/xinglu/LLMs/Qwen2-7B-Instruct'):
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tker = tokenizer
    
    def get_novel(self):
        return ""
    
    def get_qas(self):
        res=[]
        for p in self.prompts: 
            chat = [{"role": "system", "content": self.prompts_template},
                    {"role": "user", "content": p}] 
            # TODO this not works for Mistral
            chat_str = self.tker.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True)
            
            res.append({
                "Question": p,
                "prompt": chat_str
            })
        for p in self.prompts:
            res.append({
                "Question": p,
                "prompt": p
            })
        return res 

# sys_prompt = "You are a helpful assistant, who always provide explanation. Think like you are answering to a five year old."
# prompt = "What kind of task would test someone's ability to perform physical reasoning?"

# prefix = "<|im_start|>"
# suffix = "<|im_end|>\n"
# sys_format = prefix + "system\n" + sys_prompt + suffix
# user_format = prefix + "user\n" + prompt + suffix
# assistant_format = prefix + "assistant\n"
# input_text = sys_format + user_format + assistant_format

# messages = [
# {"role": "user", "content": input_text},
# ]

# encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

# model_inputs = encodeds.to(device)
# model.to(device)

# generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
# decoded = tokenizer.batch_decode(generated_ids)
# print(decoded[0])

class SantiQA(BasicQA):
    prompts=[
         # "In the third book of the Three-Body series by Liu Cixin -- Death's End. Yun Tianming told 3 fairy tales. What is the name of the first tale of Yun Tianming?",
        
        "List the names of Yun Tianming's tales.",
        'Summarize the first tale of Yun Tianming.',  
        "In Yun Tianming's tales, by He'ershingenmosiken, from linguistic aspects, what ciphered information he want to convney?", 
        
        "列出云天明故事的名字。",
        "总结云天明给程心讲的第一个故事。",
        "在云天明的故事中，多次提到赫尔辛基默斯肯这个地点，他想传递怎么信息？",
        "总结云天明的生平。",
        "叶文洁的性格是怎样的？",
        "叶文洁身上发生了什么事情使她回复了三体人的信息？",

        "Summarize Yun Tianming's life.",
        "What is Ye Wenjie's personality like?",
        "What events happened to Ye Wenjie that led her to respond to the Trisolarans?",

        "Luo Ji is a ",
    ]
    prompts_template="你是一个助手，用户会提出一些关于《三体》的问题或请求，请回答问题或满足请求。"

class FanrenQA(BasicQA):
    prompts=[
        "补全这句话：韩立是一个...。",
        
        "请介绍凡人修仙传的主线剧情。",
        "请介绍韩立的性格。",
        "请总结韩立在七玄门的经历。",
        "请总结韩立在乱星海的经历。",
        "请总结韩立与极炫的关系。",
        "请总结韩立与南宫婉的关系。",
        
         # 故事情节分析
        "《凡人修仙传》中的主线故事情节是如何展开的？有哪些关键转折点？",
        "在《凡人修仙传》中，韩立的修仙之路有哪些重要的里程碑事件？这些事件如何影响了他后续的修行？",
        "《凡人修仙传》中有哪些伏笔在后期得到了呼应？这些伏笔如何影响了故事的整体发展？",
        
        # 人物关系分析
        "韩立与南宫婉之间的感情发展是如何描写的？他们的关系对故事的推进起到了什么作用？",
        "在韩立的修行过程中，哪些人物对他影响最大？这些人物之间的关系是如何演变的？",
        "《凡人修仙传》中，正邪两派的势力斗争对人物关系有何影响？有哪些主要的矛盾和联盟？",
        
        # 人物性格分析
        "韩立的性格特点有哪些？这些特点如何帮助或阻碍了他的修仙之路？",
        "南宫婉作为女主角，她的性格与韩立有何不同？她的行事风格如何影响了她的命运？",
        "《凡人修仙传》中的反派角色有哪些？他们的性格特点和行动动机如何推动了故事情节的发展？",
        
        # 总结类问题
        "《凡人修仙传》中对修仙世界的描写有哪些独特之处？这些描写对小说的魅力有何贡献？",
        "韩立的修仙之路有哪些值得借鉴的经验和教训？这些经验对现实生活有何启示？",
        "《凡人修仙传》中的修仙理念和价值观与传统的修仙小说有何不同？这些差异如何影响了读者的阅读体验？",
        
        '''请根据以下框架，扩写故事。
1. 进入乱星海：
   - 韩立在突破瓶颈后，离开了天南，进入了乱星海。这是一个由无数岛屿组成的区域，各大势力错综复杂。
   
2. 探索机缘：
   - 韩立在乱星海中不断寻找机缘，提升自己的修为。他参与了许多秘境探险，获得了大量的修炼资源和法宝。

3. 突破境界：
   - 经过在乱星海的种种历练，韩立不断突破自身的境界，从元婴期逐步提升到更高的层次，为后续的故事发展奠定了基础。

4. 离开乱星海：
   - 最后，韩立在乱星海中完成了一定的修炼目标，准备前往新的区域继续修行。''',
    ]
    prompts_template="你是一个助手，用户会提出一些关于《凡人修仙传》的问题或请求，请用中文回复用户。"

class NovelQADS(LongContextDSBase):
    def __init__(self, option, base_path='/home/ma-user/work/xinglu/novelqa.github.io/Demonstration/'):
        self.base_path = base_path
        self.novels_path = os.path.join(base_path, 'Raw_Novels')
        self.data_path = os.path.join(base_path, 'Data')
        self.option=option

    def get_candidates(self):
        # 获取所有小说的文件名，去除扩展名，返回列表
        file_paths = glob.glob(os.path.join(self.novels_path, '*.txt'))
        options = [os.path.splitext(os.path.basename(fp))[0] for fp in file_paths]
        return options

    def get_novel(self):
        # 读取指定小说的全文
        with open(os.path.join(self.novels_path, f'{self.option}.txt'), 'r') as file:
            return file.read()

    def get_qas(self):
        # 加载指定小说的问题和答案数据
        with open(os.path.join(self.data_path, f'{self.option}.json'), 'r') as file:
            return json.load(file)

class Bamboo(LongContextDSBase):
    def __init__(self, 
                 option,
                 base_path='/home/ma-user/work/xinglu/BAMBOO/datasets',
                 must_have = r"(paper|meeting)qa_", 
                 ):
        self.base_path = base_path
        self.must_have=must_have 
        self.option = option 
        
        self.dfs={} 
        _=self.get_candidates()
    
    def get_candidates(self):
        for p in glob.glob(os.path.join(self.base_path, '*.jsonl')):
            p2=os.path.splitext(os.path.basename(p))[0]
            if not re.match(self.must_have, p2): 
                continue 
            try:
                ls=load_jsonl(p) 
            except Exception as e:
                my_prt(p, 'err', e) 
                continue
            
            df=pd.DataFrame(ls) 
            df=df.rename(columns=RENAME_MAP)
            df['content_md5'] = df.content.apply(calculate_md5)
            min_len = min_unique_length(df['content_md5'])
            df['content_md5'] = df.content_md5.apply(lambda x: x[:min_len])
            for md5, df_ in df.groupby('content_md5'): 
                if df_.shape[0]==0:continue 
                self.dfs[f'{p2}/{md5}']=df_
                
            # my_prt('INFO', p, df.columns)
            
        return list(self.dfs.keys())
    
    def get_novel(self):
        return self.dfs[self.option]['content'].iloc[0].strip()
    
    def get_qas(self):
        df=self.dfs[self.option]
        res=[] 
        for now in df.to_dict(orient='records'):
            num_op = len(now['Options'])
            if num_op==1: continue 
            now['op_str']=self.convert_choices_to_str(now['Options'])
            now["prompt"]=PROMPT_ANS_EXP.format(question=now['Question'],
                                                options=now['op_str'])  
            res.append(now) 
        return res

class PretrainFromTxt(LongContextDSBase):
    def __init__(self, data_path):
        txt_files = glob.glob(os.path.join(data_path, '*.txt')) 
        self.novels=[]
        for fn in txt_files:
            with open(fn, 'r') as f:
                self.novels.append(f.read())
    def get_novel(self):
        return self.novels 
    def get_qas(self):
        return {}

class DSFromJson(LongContextDSBase):
    def __init__(self,
                data_path, 
                split='train+val'
                 ):
        json_files = glob.glob(os.path.join(data_path, '*.json'))
        assert len(json_files)>=1, "expecting at least 1 instr-following data"
        tmp=[] 
        for json_file in json_files:
            tmp.append(pd.read_json(json_file))
        qas=pd.concat(tmp).reset_index(drop=True)
        qas=qas.rename(columns=RENAME_MAP)
        qas['Gold'] = qas['Gold'].apply(otherutils.extract_letter)
        qas.drop_duplicates(subset='Question', keep='last', inplace=True) 
        self.dfs=dict(default=qas)
        self.split=split 
        
    def get_candidates(self):
        return list(self.dfs.keys())
    
    def get_qas(self):
        res=self.dfs['default'].to_dict(orient='records')  
        num=len(res)
        splt_at= num//10*7
        start=0;end=num
        if not 'train' in self.split: 
            start = splt_at
        if not 'val' in self.split:
            end = splt_at
        
        r2=[]
        for now in res[start:end]:
            num_op = len(now['Options'])
            if num_op==1: continue 
            now['prompt'] = PROMPT_ANS_EXP.format(question=now['Question'],
                                                options=self.convert_choices_to_str(now['Options'])) 
            r2.append(now) 
        return r2
    
    def get_novel(self): 
        my_prt('Warn: SFT data not need novel') 
        return ""

def special_split(s):
    # Define the separators: space and newline
    separators = r'(\s|\n)'
    # Use re.split to split the string and keep the separators
    parts = re.split(separators, s)
    return parts

class HaystackTestDS(LongContextDSBase):
    def __init__(self, insert_place=0.5, context_len=1e3):
        self.haystack = special_split(Bamboo('paperqa_16k/69').get_novel()) 
        hs_len = len(self.haystack)
        print("hs len (not count space newline)", hs_len//2) 
        self.needle="Mary is a gentle and fashionable girl. Mary’s favorite fashion designer was Coco Chanel when she was a teenager."
        self.question = "Who was Mary’s favorite fashion designer when she was a teenager?"
        self.gold="Coco Chanel" 
        self.insert_place=insert_place 
        self.context_len=int(min(context_len*2, hs_len))
    
    def get_novel(self):
        hs=self.haystack[:self.context_len]  
        # Calculate the index where to insert the needle
        insert_index = int(len(hs) * self.insert_place)
        # Insert the needle in the calculated position
        res="".join(hs[:insert_index]).strip() +" "+ self.needle +" "+ "".join(hs[insert_index:]).strip()
        return res.strip()
    
    def get_qas(self):
        return [{
            "Question": self.question,
            "prompt": "You are a helpful assistant. Please first answer user's question, and then give a concise explanation. "+self.question, 
            # "Gold": self.gold,
            # "Options": [
            #     "(A) Coca Cola Company", 
            #     "(B) CoCo Fresh Tea", 
            #     "(C) Coco Channel", 
            #     "(D) Yves Saint Laurent", 
            #     "(E) Christian Dior"
            # ] # not used
        }]
    
class HaystackRelateToNeedle(HaystackTestDS):
    def __init__(self, insert_place=0.5, context_len=1000):
        haystake='''Mary has always been recognized as a gentle and fashionable individual, traits that have defined her character since her early teenage years. Her affinity for fashion is not just about clothing; it is a profound expression of her personality and emotions. As a young girl, Mary found herself mesmerized by the elegance and boldness of Coco Chanel's designs. Chanel, a pioneering figure in the fashion industry, was renowned for her revolutionary approach to women's wear, emphasizing comfort alongside style, a philosophy that resonated deeply with Mary.

Growing up in a small, picturesque town, Mary’s early exposure to high fashion was limited to glossy magazines and the occasional runway show broadcast on television. These windows into the world of haute couture captivated her imagination and played a significant role in shaping her aspirations. She would spend hours sketching designs in her notebook, each line a step closer to her dreams, each sketch a tribute to Chanel's influence. Mary admired how Coco Chanel had overcome numerous obstacles to redefine the standards of beauty and fashion, which inspired her to pursue a path where she could also make a mark.

Mary’s teenage years were a mosaic of personal growth and discovery. She joined the fashion club at her high school, where she not only honed her design skills but also learned the intricacies of fabric selection, sewing techniques, and the importance of tailoring. Her gentle demeanor made her a beloved figure among her peers, while her distinctive style set her apart at school events and gatherings. She wasn't just following trends; she was learning to create them.

Chanel's famous quote, "Fashion fades, only style remains the same," became Mary's mantra. She began experimenting with her wardrobe, combining timeless pieces in ways that exuded sophistication and simplicity—hallmarks of Chanel’s influence. Mary’s approach to fashion was not about lavish expenditures but about making a statement with simplicity and grace. Her ability to pull off a look that carried the whisper of vintage elegance with a modern twist soon became her signature.

As she progressed through her teenage years, Mary's reputation as a fashionista grew. Her blog, "Mary’s Vogue Voyage," started as a small project to document her daily outfits but soon attracted a dedicated following. Readers were drawn to her gentle narrative voice and her insightful commentary on how fashion could be both a personal and a political statement, reflecting one’s identity and values. Through her blog, Mary explored the impact of fashion icons like Coco Chanel, who not only transformed women’s wardrobes but also their roles in society.

By the time Mary graduated from high school, she had transformed from a fashion enthusiast into an emerging influencer in the field. Her understanding of fashion as an art form and a means of expression had deepened, driven by her early fascination with Chanel’s legacy. Mary’s journey into the world of fashion was not just about personal style but about inspiring others to find their unique voice through the clothes they wore. As she prepared to step into the broader world, her dreams were clearer than ever, fueled by the elegance and empowerment that Coco Chanel had epitomized.

As Mary transitioned into her collegiate years, her passion for fashion followed her to university, where she chose to major in Fashion Design. This decision marked the beginning of a new chapter, one that brought her closer to her dreams of becoming a significant figure in the fashion world. University life offered Mary a broader platform to showcase her talent and creativity. She immersed herself in the study of fashion history, textiles, and fashion merchandising, each course adding layers to her understanding and appreciation of the industry.

At university, Mary also encountered a vibrant community of like-minded individuals, each with their own unique styles and visions. This environment fostered collaboration and innovation, and Mary thrived in it. She became involved in various fashion shows organized by the university, often taking the lead in designing costumes and coordinating runway setups. These experiences were pivotal, providing her with real-world insights into the logistical and creative challenges of fashion show production. Her ability to blend classic aesthetics with contemporary trends received accolades, and her designs often stood out for their elegance and wearability.

Internships with local fashion houses were a stepping stone that brought Mary closer to the practical aspects of the fashion industry. These stints provided her with the opportunity to work alongside seasoned designers and business managers, learning the nuances of fashion from conceptualization to consumer. One of her most cherished experiences was an internship at a boutique that specialized in reviving vintage styles for modern wear, a concept that resonated deeply with Mary’s admiration for Coco Chanel.

During her time at university, Mary also expanded her digital footprint by transforming her blog into a more comprehensive fashion portal. She started producing content that not only highlighted her daily fashion picks but also included interviews with emerging designers, reviews of fashion weeks around the world, and discussions on sustainable fashion—a topic she grew increasingly passionate about. Her platform became a space for advocating for ethical fashion practices, echoing Chanel’s commitment to innovation and breaking norms.

As her blog gained traction, Mary started receiving invitations to speak at fashion events and participate in panel discussions. These engagements enhanced her profile and connected her with professionals across the fashion industry. Her perspectives on fashion as a form of personal and cultural expression were well-received, and she quickly became known as a fresh voice in fashion commentary, one that combined respect for historical influences with a keen eye for future trends.

Mary's university years were not only about building her career but also about personal exploration and establishing her identity within the fashion world. She experimented with various styles, always infusing her creations with a sense of history and narrative. Her collections told stories—of eras, movements, and icons, all while staying true to her belief in fashion as an accessible art form. Her gentle approach and sophisticated designs continued to garner attention, setting the stage for the final leg of her journey towards becoming a professional designer.

Upon completing her degree in Fashion Design, Mary was at a crossroads, equipped with a wealth of knowledge, experience, and a burgeoning reputation in the fashion community. Her final thesis, a collection inspired by Coco Chanel’s early works but reimagined for the contemporary woman, was a resounding success, captivating both her professors and visitors at the graduate showcase. This collection not only served as a homage to her lifelong inspiration but also as a declaration of her unique creative vision.

Mary's success at the graduate showcase led to job offers from several prestigious fashion labels, but she chose to start her own brand. This decision stemmed from her desire to retain creative control and continue pushing the boundaries of fashion with her distinctive blend of historical elegance and modern simplicity. She founded her label with a commitment to sustainability, an issue that had become increasingly important to her over the years.

In the early days of her venture, Mary faced numerous challenges, from sourcing ethically produced materials to finding the right collaborators who shared her vision and values. She remained undeterred, driven by her passion and the knowledge that her brand stood for something significant—fashion that was both beautiful and responsible. Her collections were characterized by minimalist designs, impeccable tailoring, and a strong emphasis on versatility and sustainability.

Mary’s brand quickly gained recognition not only for its aesthetic appeal but also for its commitment to ethical practices. Her online store expanded her reach globally, allowing her to connect with customers worldwide who were seeking fashion that made a statement in more ways than one. Mary also leveraged social media to tell her brand’s story, from behind-the-scenes insights into her creative process to spotlighting the artisans who brought her designs to life.

As her business grew, Mary continued to engage with the fashion community, often returning to her alma mater to mentor young designers. Her talks and workshops focused on the importance of personal vision and ethical responsibility in fashion, inspiring the next generation to think differently about the clothes they designed and wore.

Mary’s journey from a fashion-obsessed teenager influenced by Coco Chanel to a respected designer with her own label was marked by numerous achievements and learning experiences. Her commitment to maintaining her style while adhering to sustainable practices set her apart in the fashion industry, making her a role model for aspiring designers everywhere.

Years later, as she reflected on her career, Mary realized that her success was not just due to her talent or hard work but also her ability to stay true to her values. Fashion, for Mary, was always more than just clothing; it was about making a statement, pushing boundaries, and, most importantly, staying true to oneself. She had not only followed in the footsteps of her idol Coco Chanel but also carved out a path that was distinctly her own, influencing the industry in profound ways. Through her gentle but determined approach, Mary continued to inspire and shape the world of fashion, just as Chanel had done decades before her.

Mary's journey in the fashion industry exemplifies how passion, perseverance, and principles can lead to profound success and influence. From her early days as a teenager inspired by the legendary Coco Chanel to becoming a luminary in her own right, Mary's story is a testament to the power of staying true to one’s ethos in a competitive and ever-changing field. Her blend of historical appreciation with a commitment to sustainability has not only set her apart but also paved the way for future generations of designers. By aligning her business practices with her personal values, Mary has shown that fashion can be both beautiful and benevolent, influencing societal trends and consumer behaviors worldwide. Her legacy, underscored by her gentle demeanor and formidable resolve, continues to inspire a dialogue about the role of fashion in expressing identity and effecting social change. As Mary’s brand grows, it remains a beacon of integrity and innovation, proving that true style is timeless and that real change is possible when you dare to wear your values on your sleeve.'''
        self.haystack = special_split(haystake) 
        self.context_len=int(min(context_len*2, len(self.haystack)))
        self.needle="Mary is a gentle and fashionable girl. Mary’s favorite fashion designer was Coco Chanel when she was a teenager."
        self.question = "Who was Mary’s favorite fashion designer when she was a teenager?"
        self.gold="Coco Chanel" 
        self.insert_place=insert_place 
        print("hs len (not count space newline)", len(self.haystack)//2) 
        

if __name__=="__main__":
    from transformers import AutoTokenizer
    # tkr=AutoTokenizer.from_pretrained('/home/ma-user/work/xinglu/LLMs/Mistral-7B-Instruct-v0.3')
    tkr=AutoTokenizer.from_pretrained('/home/ma-user/work/xinglu/LLMs/llama-3-8b-ins')

    # ds=HaystackRelateToNeedle(0.5, 2e3)
    # print( len(ds.needle), len(tkr.tokenize(ds.get_novel()) ) )
    
    ds=DSFromJson(
        data_path="/home/ma-user/work/xinglu/sq-gptq/llm_pretrain/data/meetingqa_16k/2844",
        # data_path = '/home/ma-user/work/xinglu/sq-gptq/llm_pretrain/data/paperqa_16k/69/',
        # data_path="/home/ma-user/work/xinglu/sq-gptq/llm_pretrain/data/frank",
        split='train'
    )
    # print(len(ds.get_qas()), len(ds.get_permed_qas()))
    ds=PretrainFromTxt('/home/ma-user/work/xinglu/sq-gptq/llm_pretrain/data/full')
    print(ds.get_novel())
    
    # ds=NovelQADS('Frankenstein')
    # ds=Bamboo('meetingqa_16k/2844')
    # qas = ds.get_permed_qas() 
    
    # for c in ds.get_candidates():
    #     book = Bamboo(c).get_novel() 
    #     tks = tkr.tokenize(book) 
    #     df=ds.dfs[c]
    #     print(c, len(tks), df.shape)
    
    # embed() 

