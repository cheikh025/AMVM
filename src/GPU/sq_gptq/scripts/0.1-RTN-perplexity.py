import sys
import os
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from modelutils import TextGenerator, DEV
from transformers import AutoTokenizer, TextGenerationPipeline
import netutils 
from datautils import get_loaders
from smoothquant import get_act_scales, smooth_lm 

nSentence, tokenPerSentence=128, 2048 # 128
MDL="facebook/opt-125m"
# MDL="/home/ma-user/work/xinglu/LLMs/llama-2-7b-hf" # 
for DS in ["c4"]: # "wikitext2" ,
    with netutils.no_ssl_verification():
        train_loader, val_loader = get_loaders(DS, model=MDL, nsamples=nSentence) 
        opt_generator = TextGenerator( MDL )
    model=opt_generator.model.half()
    tokenizer = opt_generator.tokenizer 

    # act_scales = get_act_scales(model, tokenizer, train_loader, num_samples=len(train_loader))
    # smooth_lm(model, act_scales, 0.5)

    ## sq_path='output/opt-125m-sq'
    ## model.save_pretrained(sq_path)

    from modelutils import replace_linear_with_RNT_linear
    ## naive proj to INT grid 
    replace_linear_with_RNT_linear(model, xbit=16, wbit=3) 

    from ppl_eval import Evaluator
    evaluator = Evaluator(val_loader, tokenizer, DEV, )
    ppl = evaluator.evaluate(model)
    print(f"{DS} Perplexity: {ppl}")



