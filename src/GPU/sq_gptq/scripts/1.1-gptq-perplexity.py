import sys
import os
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
from modelutils import TextGenerator, DEV, ExecutionOrderTracker, LayerDataCache
from transformers import AutoTokenizer, TextGenerationPipeline, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from auto_gptq.modeling.opt import OPTGPTQForCausalLM
import netutils 
from datautils import get_loaders, save_tensors_to_hdf5
from auto_gptq.modeling._base import BaseGPTQForCausalLM 
# from gptq_patch import quantize_monkey_patch

# BaseGPTQForCausalLM.quantize = quantize_monkey_patch

nSentence, tokenPerSentence=128, 2048 # 128
MDL="facebook/opt-125m"
# MDL="/home/ma-user/work/xinglu/LLMs/llama-2-7b-hf" # 
MDL_NM=os.path.basename(MDL) 
for DS in [ "c4",]: # "wikitext2" ,
    train_loader, val_loader = get_loaders(DS, model=MDL, nsamples=nSentence) 

    quantized_model_dir = f"output/{MDL_NM}-gptq"
    quantize_config = BaseQuantizeConfig(
        bits=3, 
        group_size=-1,  # 一般推荐将此参数的值设置为 128
        desc_act=False,  # 设为 False 可以显著提升推理速度，但是 ppl 可能会轻微地变差
        sym=1, #False
    )

    # 加载未量化的模型，默认情况下，模型总是会被加载到 CPU 内存中
    model = AutoGPTQForCausalLM.from_pretrained(MDL, quantize_config)
    with netutils.no_ssl_verification():
        tokenizer = TextGenerator( MDL ).tokenizer 
    # Annotate the model variable
    model: OPTGPTQForCausalLM

    # 量化模型, 样本的数据类型应该为 List[Dict]，其中字典的键有且仅有 input_ids 和 attention_mask
    fwei = model.quantize(train_loader)
    if fwei: save_tensors_to_hdf5(fwei, 'output/gptq_asym_weights.h5')
    
    # 保存量化好的模型
    model.save_quantized(quantized_model_dir, use_safetensors=True)
    model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device=DEV)
    model.to(DEV)

    ## for opt model
    # t=model.model.model.decoder.layers[0].self_attn.k_proj
    # from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear
    # t: QuantLinear

    from modelutils import replace_linear_with_RNT_linear

    ## naive proj to int8 grid 
    # replace_linear_with_RNT_linear(model, xbit=8, wbit=4)   

    from ppl_eval import Evaluator
    evaluator = Evaluator(val_loader, tokenizer, DEV, )
    ppl = evaluator.evaluate(model)
    print(f"{DS} Perplexity: {ppl}")

