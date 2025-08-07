from utils.modelutils import DEV
import torch.nn as nn
from transformers import AutoTokenizer
import torch
import tqdm
import argparse

class Evaluator:
    def __init__(self, dataset, tokenizer, device, ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.n_samples = len(dataset)

    @torch.no_grad()
    def evaluate(self, model, verbose=0):
        model.eval()
        nlls = []
        model_dev = next(model.parameters()).device
        for i in tqdm.tqdm(range(self.n_samples), desc="Evaluating..."):
            batch = self.dataset[i]['input_ids'].to(model_dev)
            with torch.no_grad():
                lm_logits = model(batch).logits
            
            if i==0 and verbose:
                input_text = self.tokenizer.decode(batch[0], skip_special_tokens=True)
                sample_output_ids = torch.argmax(lm_logits, dim=-1)
                sample_output_text = self.tokenizer.decode(sample_output_ids[0], skip_special_tokens=True)
                print('=='*39, '\n', f"Sample {i} Input: {input_text}", flush=True)
                print('=='*39, '\n', f"Sample {i} Output: {sample_output_text}", flush=True)
            
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = batch[:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / (self.n_samples * 2048))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, 
                        default='facebook/opt-125m'
                        )

    args = parser.parse_args()
    model_path = args.model_path
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    from src.GPU.utils.datautils import get_loaders
    trainloader, dataset=get_loaders("c4", model="facebook/opt-125m", nsamples=40)
    # trainloader, dataset=get_loaders("ptb", model="facebook/opt-125m", nsamples=40)
    evaluator = Evaluator(dataset, tokenizer, DEV, )

    ## smoothquant+gptq
    from auto_gptq import AutoGPTQForCausalLM
    model_path='/home/ma-user/work/xinglu/sq-gptq/output/gptq_asym_weights.h5' #opt-125m-8b'

    model = AutoGPTQForCausalLM.from_quantized(model_path, device=DEV)

    ppl = evaluator.evaluate(model)
    print(f"Perplexity: {ppl}")
