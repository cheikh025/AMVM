import sys
import os
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from netutils import * 
from datautils import *  
from modelutils import TextGenerator

nSentence, tokenPerSentence=128, 2048 # 128
with no_ssl_verification():
    loader, _ = get_loaders("c4", model="facebook/opt-125m", nsamples=nSentence) 
    # loader, valenc = get_loaders("wikitext2", model="facebook/opt-125m")
    # loader, valenc = get_loaders("ptb", model="facebook/opt-125m")
# For OPT model
# opt_generator = TextGenerator("facebook/opt-1.3b", )
    opt_generator = TextGenerator("facebook/opt-125m", )
# print(opt_generator.generate_text("Once upon a time"))
# # For BLOOM model
# # bloom_generator = TextGenerator(model_name="bigscience/bloom-560m")

model=opt_generator.model
tokenizer = opt_generator.tokenizer 
from modelutils import ExecutionOrderTracker 
tracker = ExecutionOrderTracker(model, opt_generator.tokenizer) 
layer_nms= tracker.get_layer_names() 

from modelutils import LayerDataCache 
layer_cache = LayerDataCache(model, ) 
layer_cache.rehook(layer_nms)
layer_cache.process_inputs(loader) 
# model.model.decoder.layers[0].self_attn.k_proj.weight 
X,W=layer_cache.get_caches()

keys = set(X.keys()).union(W.keys())
for k in sorted(keys):
    assert k in X, k
    assert k in W, k
    print(f"X[{k}]:", X[k].shape, f"W:", W[k].shape)

keys = list(X.keys())
mid_index = len(keys) // 2

# Splitting the dictionary into two parts
X1 = {key: X[key] for key in keys[:mid_index]}
X2 = {key: X[key] for key in keys[mid_index:]}
layer_cache.save_tensors_to_hdf5(X1, 'output/cached_inputs_part1.h5')
layer_cache.save_tensors_to_hdf5(X2, 'output/cached_inputs_part2.h5')
# layer_cache.save_tensors_to_hdf5(X,'output/cached_inputs.h5')

layer_cache.save_tensors_to_hdf5(W,'output/cached_weights.h5') 


print("ok")


