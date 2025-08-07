"""
Class to perform perplexity evaluations for quantized weights
"""

import sys
import os
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from utils.modelutils import TextGenerator, DEV
from transformers import AutoTokenizer, TextGenerationPipeline
from utils.netutils import no_ssl_verification
from utils.datautils import get_loaders, load_tensors_from_hdf5
from utils.modelutils import LayerDataCache
from utils.ppl_eval import Evaluator
import utils.utils as ut
import torch

from utils.utils import get_dequantized_squeezellm_model

# from SqueezeLLM.llama import load_quant
import numpy as np


# TODO: remove, this was for testing purposese
def test_rtn_squeezellm(model, state_dict: dict[str, torch.Tensor]):
    squeezellm_LUT = ut.get_squeezellm_lookup_tables(model, state_dict)

    for name, module in model.named_modules():
        if 'lm_head' in name:
            continue

        if isinstance(module, torch.nn.Linear):
            # replace with RTN model
            LUT = squeezellm_LUT[name]
            new_tensor = torch.zeros_like(module.weight.data)
            for row in range(new_tensor.shape[0]):
                new_tensor[row] = LUT[row][ut.round_to_nearest_pole_sim(module.weight.data[row], LUT[row], device=torch.device('cpu'))]
            module.weight.data = new_tensor
            print(f"Round to nearest for {name}")


    return model



@torch.no_grad()
def evaluate_perplexity(base_model: str, datasets: list[str], quantized_weights_path: str, use_calibration_dataset: bool=True) -> None:
    """
    Prints out the perplexity of the weights based off of these arguments:
    Arguments:
        - base_model: path to the base model (what model the original weights were from)
        - datasets: dataset names to evaluate perplexity on (c4 or wikitext2)
        - quantized_weights_path: path to the quantized weights
        - use_calibration_dataset: if True, then we evaluate perplexity on the calibration dataset, otherwise we evaluate it on
        the eval dataset

    Replaces all the weights in the base model with the quantized weights, then evaluates perplexity on the model.
    """


    nSentence, tokenPerSentence = 128, 2048  # 128
    with no_ssl_verification():
        opt_generator = TextGenerator(base_model)
        model = opt_generator.model.half()
    tokenizer = opt_generator.tokenizer

    print(model)
    model.to("cuda:0")


    print(f"Using weight path {quantized_weights_path}")
    W = load_tensors_from_hdf5(quantized_weights_path)

    layer_cache = LayerDataCache(model, )
    # Replace the model weights with quantized weights
 #   for nm, w in W.items():
 #       w = torch.from_numpy(w).half()
 #       layer_cache.set_model_weights(nm, w)
 #       print(f"Replaced layers {nm}")
#
 #   print(f"Replaced layers")


    # Evaluate perplexity on the datasets
    for DS in datasets:
        print(f"Loading dataset {DS}")
        with no_ssl_verification():
            train_loader, val_loader = get_loaders(DS, model=base_model, nsamples=nSentence)
            print("Loaded eval data")
        # To save the inputs going through the model, uncomment code below
        # -------------------------------------------------------------------------
        # input_save_path = 'aug21_from_squeezellm.pt'
        # from utils.modelutils import ExecutionOrderTracker
        # tracker = ExecutionOrderTracker(model, opt_generator.tokenizer)
        # layer_nms = tracker.get_layer_names()
        #
        # layer_cache.rehook(layer_nms)
        # layer_cache.process_inputs(train_loader)
        #
        # print("Done processing inputs, now saving: ")
        # torch.save(layer_cache.cached_inputs, input_save_path)
        # -------------------------------------------------------------------------

        if use_calibration_dataset:
            evaluator = Evaluator(train_loader, tokenizer, DEV)
        else:
            evaluator = Evaluator(val_loader, tokenizer, DEV)
        ppl = evaluator.evaluate(model)
        print(f"{DS} Perplexity: {ppl}")



if __name__ == '__main__':
    # MDL="/home/ma-user/work/xinglu/LLMs/llama-2-7b-hf"
    MDL = "./FB/"#"facebook/opt-125m"

    datasets = ["c4"]  # c4 or wikitext2. Ex: ["c4", "wikitext2"] will evaluate c4 first then wikitext2
    quantized_weights_path = "aug21_squeezellm_rtn.h5"
    evaluate_perplexity(MDL, datasets, quantized_weights_path, use_calibration_dataset=False)




