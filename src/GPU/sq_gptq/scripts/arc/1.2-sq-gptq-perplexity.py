import sys
import os
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from modelutils import TextGenerator, DEV
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import netutils 
from datautils import get_loaders

nSentence, tokenPerSentence=128, 2048 # 128
MDL="facebook/opt-125m"
MDL="/home/ma-user/work/xinglu/LLMs/llama-2-7b-hf" # 
MDL_NM=os.path.basename(MDL) 
train_loader, val_loader = get_loaders("wikitext2", model=MDL, nsamples=nSentence) 

opt_generator = TextGenerator(MDL)
model=opt_generator.model
tokenizer = opt_generator.tokenizer 

from smoothquant import get_act_scales, smooth_lm 

act_scales = get_act_scales(model, tokenizer, train_loader, num_samples=len(train_loader))
smooth_lm(model, act_scales, 0.5)
sq_path=f'output/{MDL_NM}-sq'
model.save_pretrained(sq_path)
quantized_model_dir = f"output/{MDL_NM}-sq-gptq"
quantize_config = BaseQuantizeConfig(
    bits=4, 
    group_size=-1,  # 一般推荐将此参数的值设置为 128
    desc_act=False,  # 设为 False 可以显著提升推理速度，但是 ppl 可能会轻微地变差
)

# 加载未量化的模型，默认情况下，模型总是会被加载到 CPU 内存中
model = AutoGPTQForCausalLM.from_pretrained(sq_path, quantize_config)

# 量化模型, 样本的数据类型应该为 List[Dict]，其中字典的键有且仅有 input_ids 和 attention_mask
model.quantize(train_loader)

# 保存量化好的模型
model.save_quantized(quantized_model_dir, use_safetensors=True)
model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device=DEV)

from modelutils import replace_linear_with_RNT_linear

## naive proj to int8 grid 
replace_linear_with_RNT_linear(model, xbit=8, wbit=4)   


from ppl_eval import Evaluator
evaluator = Evaluator(val_loader, tokenizer, DEV, )
ppl = evaluator.evaluate(model)
print(f"Perplexity: {ppl}")