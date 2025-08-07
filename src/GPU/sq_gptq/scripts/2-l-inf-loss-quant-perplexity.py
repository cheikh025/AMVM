import sys
import os
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from modelutils import TextGenerator, DEV
from transformers import AutoTokenizer, TextGenerationPipeline
from netutils import no_ssl_verification
from datautils import get_loaders, load_tensors_from_hdf5
from modelutils import LayerDataCache 
from ppl_eval import Evaluator
import torch 

nSentence, tokenPerSentence=128, 2048 # 128
# MDL="/home/ma-user/work/xinglu/LLMs/llama-2-7b-hf" 
MDL="facebook/opt-125m"
with no_ssl_verification():
    opt_generator = TextGenerator( MDL )
    model=opt_generator.model.half()
tokenizer = opt_generator.tokenizer 

W=load_tensors_from_hdf5('output/merged_quantized_weights.h5') 
# W=load_tensors_from_hdf5('output/gptq_weights.h5') 

layer_cache = LayerDataCache(model, ) 
for nm, w in W.items():
    w=torch.from_numpy(w).half()
    layer_cache.set_model_weights(nm, w) 

for DS in ["c4", ]: #   "wikitext2" , 
    with no_ssl_verification():
        train_loader, val_loader = get_loaders(DS, model=MDL, nsamples=nSentence) 
        
    evaluator = Evaluator(val_loader, tokenizer, DEV)
    ppl = evaluator.evaluate(model)
    print(f"{DS} Perplexity: {ppl}")

