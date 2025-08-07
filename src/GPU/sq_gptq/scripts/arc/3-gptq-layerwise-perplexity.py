
# refer to 
# from auto_gptq.modeling._utils import quantize_dequantize
# from auto_gptq.modeling._base import * 
import sys
import os

# Get the absolute path of the current script
script_path = os.path.abspath(__file__)

# Get the directory containing the script
script_dir = os.path.dirname(script_path)

# Get the parent directory of the script's directory
parent_dir = os.path.dirname(script_dir)

# Insert the parent directory into the system path
sys.path.insert(0, parent_dir)
from quant import Quantizer
from gptq_patch import GPTQ
import torch.nn as nn 
import torch
dev='npu:1'
layer=nn.Linear(1920, 5120, bias=False).to(dev)
gptq={}
name='fake_linear' 
gptq[name]=GPTQ(layer) 
gptq[name].quantizer = Quantizer() 
gptq[name].quantizer.configure(bits=4, perchannel=True) 
gptq[name].name=name
def add_batch(name):
    def tmp(_, inp, __):
        gptq[name].add_batch(inp[0].data)
    return tmp
handles = []
handles.append(layer.register_forward_hook(add_batch(name)))
_=layer(
        torch.randn(10,1920).half().to(dev)
        )
gptq[name].fasterquant(
)


from modelutils import LayerDataCache 
from modelutils import TextGenerator
from datautils import get_loaders
loader, valenc = get_loaders("c4", model="facebook/opt-125m", nsamples=2)

opt_generator = TextGenerator("facebook/opt-125m", )
print(opt_generator.generate_text("Once upon a time"))
model=opt_generator.model
from modelutils import ExecutionOrderTracker 
tracker = ExecutionOrderTracker(model, opt_generator.tokenizer) 
layer_nms= tracker.get_layer_names() 
layer_cache = LayerDataCache(model) 

for now_idx in range(len(layer_nms)): 
    now_layer_nm=layer_nms[now_idx] 
    layer_cache.rehook([now_layer_nm])
    layer_cache.process_inputs(loader) 
    inps, weights=layer_cache.get_caches() 
    # weights are fp32
    # inps are fp16, for saving space

    ## quantize&dequantize from different method 
    old_weights = weights[now_layer_nm]
    # print(old_weights.dtype)  --> torch.float32 
    # new_weights = quantize&dequantize(old_weights) # must be fp32 
    new_weights = old_weights 
    layer_cache.set_model_weights(now_layer_nm, new_weights) 
    
