import sys
import os
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from modelutils import TextGenerator, DEV
from transformers import AutoTokenizer, TextGenerationPipeline
from netutils import no_ssl_verification
from datautils import get_loaders

nSentence, tokenPerSentence=128, 2048 # 128
# MDL="/home/ma-user/work/xinglu/LLMs/llama-2-7b-hf" 
MDL="facebook/opt-125m"
for DS in ["c4",]: #  'ptb', "wikitext2" , 
    with no_ssl_verification():
        train_loader, val_loader = get_loaders(DS, model=MDL, nsamples=nSentence) 
        opt_generator = TextGenerator( MDL )
        
    model=opt_generator.model.half()
    tokenizer = opt_generator.tokenizer 

    from ppl_eval import Evaluator
    evaluator = Evaluator(val_loader, tokenizer, DEV)
    ppl = evaluator.evaluate(model)
    print(f"{DS} Perplexity: {ppl}")
