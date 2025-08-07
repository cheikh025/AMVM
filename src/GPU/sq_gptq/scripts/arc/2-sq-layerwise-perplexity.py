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

import torch
import torch.nn as nn 
import re

def extract_layer_info(input_string):
    # Regex pattern to match a sequence ending with ".layers.\d+"
    pattern = r'.*\.layers\.\d+'
    
    # Search using the pattern
    match = re.search(pattern, input_string)
    
    if match:
        return match.group(0)  # Returns the matched substring
    else:
        return "No match found"
    
def find_layer_by_name(model, name, acc_name=""):
    # Base case: Check if accumulated name matches the target name
    if acc_name.rstrip('.') == name:  # Strip trailing dots for accurate matching
        return model
    
    # Iterate through all submodules of the current model
    for module_name, module in model.named_children():  # Use named_children for direct submodules only
        # print(module_name)
        if acc_name:  # If there's an accumulated name, add a dot
            new_acc_name = f"{acc_name}.{module_name}"
        else:
            new_acc_name = module_name
        
        # Recursively search in the submodule
        result = find_layer_by_name(module, name, new_acc_name)
        if result is not None:
            return result
    return None

def find_LNs(layer): 
    res=[]
    for name, module in layer.named_modules():
        # Check if the module is an instance of LayerNorm
        if isinstance(module, nn.LayerNorm): # todo 
            # Process the LayerNorm module here
            # print(f"Found LayerNorm: {name}")
            res.append(module) 
    return res 

@torch.no_grad()
def extract_activation_scale(tensor):
    hidden_dim = tensor.shape[-1]
    tensor = tensor.view(-1, hidden_dim).abs()
    activation_scale = torch.max(tensor, dim=0)[0].float()
    return activation_scale


@torch.no_grad()
def smooth_ln_fcs_with_dynamic_act_scales(ln, fcs, tensor_X, alpha=0.5):
    # Ensure `fcs` is a list for uniform processing
    if not isinstance(fcs, list):
        fcs = [fcs]

    # Validate the input types
    assert isinstance(ln, (nn.LayerNorm,))  # Updated to realistic LayerNorm types  nn.LambdaLR
    for fc in fcs:
        assert isinstance(fc, nn.Linear)

    # Calculate the activation scale for tensor_X
    act_scales = extract_activation_scale(tensor_X)

    # Check dimensions match
    assert ln.weight.numel() == fc.in_features == act_scales.numel()

    # Determine the necessary properties for conversion
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    ln.to(device) # cuz for pytorch. this is inplace

    # Calculate weight scales
    weight_scales = torch.cat([fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    # Calculate final scales for normalization
    scales = (act_scales.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-5).to(device).to(dtype)

    # Adjust LayerNorm weights and biases
    ln.weight.div_(scales)
    if getattr(ln, 'bias', None) is not None:
        ln.bias.div_(scales)

    # Adjust Linear layer weights
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    return scales


def is_qkv(layer):
    return  '.q_proj' in layer or '.k_proj' in layer or '.v_proj' in layer

from modelutils import LayerDataCache 
from modelutils import TextGenerator
from datautils import get_loaders
loader, valenc = get_loaders("c4", model="facebook/opt-125m", nsamples=2)

opt_generator = TextGenerator("facebook/opt-125m", )
# print(opt_generator.generate_text("Once upon a time"))
model=opt_generator.model
from modelutils import ExecutionOrderTracker 
tracker = ExecutionOrderTracker(model, opt_generator.tokenizer) 
layer_nms= tracker.get_layer_names() 

layer_nms_grouped=[] 
for layer in layer_nms:
    if is_qkv(layer): 
        if not layer_nms_grouped or not isinstance(layer_nms_grouped[-1], list): 
            layer_nms_grouped.append([layer])
        else:
            layer_nms_grouped[-1].append(layer)
    else:
        layer_nms_grouped.append([layer])

layer_cache = LayerDataCache(model) 

for now_layer_nm in (layer_nms_grouped): 
    if not isinstance(now_layer_nm, list): now_layer_nm = [now_layer_nm]
    dlayer=find_layer_by_name(model, extract_layer_info(now_layer_nm[0]))  
    
    if not is_qkv(now_layer_nm[0]) and not 'fc1' in now_layer_nm[0]: 
        # todo
        continue
    layer_cache.rehook(now_layer_nm)
    layer_cache.process_inputs(loader) 
    inps, weights=layer_cache.get_caches() 
    # weights are fp32
    # inps are fp16, for saving space
    
    if is_qkv(now_layer_nm[0]): ln = dlayer.self_attn_layer_norm 
    else: ln = dlayer.final_layer_norm
    fcs = [tracker.mapping[nm] for nm in now_layer_nm]
    smooth_ln_fcs_with_dynamic_act_scales(ln, fcs, next(iter(inps.values())))
    
    
