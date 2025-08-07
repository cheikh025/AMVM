import os, gpustat 
def get_free_gpu(num_gpus=4):
    gpu_stats = gpustat.GPUStatCollection.new_query()
    num_gpus=min(num_gpus, len(gpu_stats))
    free_memory = [[gpu.memory_free, gpu.index] for gpu in gpu_stats]
    free_memory[0][0]-=1 # not prefer gpu 0 
    free_memory[1][0]-=1 # not prefer gpu 1 
    # Sort by free memory in descending order
    free_memory.sort(reverse=True, key=lambda x: x[0])
    
    # Get the indices of the top 'num_gpus' GPUs
    best_gpus = [free_memory[i][1] for i in range(num_gpus)]
    best_gpus = [int(gpu) for gpu in best_gpus]
    print('get dev', best_gpus)
    
    return ','.join(map(str, best_gpus))
# os.environ['CUDA_VISIBLE_DEVICES']=get_free_gpu() #'2,3'
import torch
DEV = torch.device(f'cuda:0') if torch.cuda.is_available() else 'cpu'
# DEV = torch.device('cpu')
assert torch.cuda.is_available()

import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import numpy as np 

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


class TextGenerator:
    def __init__(self, model_path_or_name, device=DEV, seq_len=2048): 
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize tokenizer and model from the provided path or model name
        self.tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, model_max_length=seq_len)
        self.model = AutoModelForCausalLM.from_pretrained(model_path_or_name,)
        self.model.eval().to(self.device)

    @torch.no_grad()
    @torch.inference_mode()
    def generate_text(self, prompt, max_length=50, seed=6):
        from otherutils import set_seed
        set_seed(seed)
        # Encode the prompt into tokens
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate a sequence of tokens following the prompt
        output = self.model.generate(**inputs, max_length=max_length)
        
        # Decode the tokens to a string
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text


class ExecutionOrderTracker:
    def __init__(self, model, tokenizer):
        self.model = model
        self.device = next(model.parameters()).device
        self.tokenizer=tokenizer
        self.execution_order = []
        self.mapping = {} 
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register a forward hook on each module"""
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Only register hooks to nn.Linear layers
                hook = module.register_forward_hook(self._capture_execution_order_hook(name))
                self.hooks.append(hook)

    def _capture_execution_order_hook(self, name):
        """Create a hook function that captures the name of the module during execution."""
        def hook(module, inp, out):
            self.execution_order.append(name)
            self.mapping[name]=module
        return hook

    def remove_hooks(self):
        """Remove all registered hooks, important for cleaning up."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_layer_names(self): 
        inputs = self.tokenizer("hi I am", return_tensors="pt")["input_ids"].to(self.device)
        # Generate a sequence of tokens following the prompt
        self.execution_order=[] 
        _ = self.model(inputs) 
        self.remove_hooks() 
        self.execution_order = [name for name in self.execution_order if 'lm_head' not in name and 'embed' not in name]
        return self.execution_order
    
    def __del__(self):
        self.remove_hooks() 
            

class LayerDataCache:
    def __init__(self, model, device=DEV, layers_to_hook=None):
        self.model = model
        self.device = device
        self.layers_to_hook = set(layers_to_hook) if layers_to_hook else None
        self.cached_weights = {}
        self.cached_inputs = defaultdict(list)
        self.execution_order = []
        self.hooks = []
        # self.init_cache()

    def init_cache(self):
        for name, layer in self.model.named_modules():
            if (isinstance(layer, nn.Linear) or 'QuantLinear' in str(layer)) and (self.layers_to_hook is None or name in self.layers_to_hook):
                self.cached_weights[name] = layer.weight.detach().cpu() # XW^T for pytorch, remember!
                # Register a forward hook to accumulate inputs
                hook = layer.register_forward_hook(self.accumulate_inputs_hook(name))
                self.hooks.append(hook)

    def accumulate_inputs_hook(self, name):
        def hook(layer, inp, out):
            if name not in self.execution_order:
                self.execution_order.append(name)  # Append the layer name on first execution
            # Append the input of the current batch to the list of inputs for this layer
            self.cached_inputs[name].append(inp[0].detach().half().cpu()) 
        return hook

    @torch.no_grad()
    def process_inputs(self, loader):
        for inp in loader:
            # todo early stop? 
            _ = self.model(inp['input_ids'].to(self.device), 
                           inp['attention_mask'].to(self.device)
                           )

    @torch.no_grad()
    def process_inputs_1batch(self, loader):
        input_ids = torch.cat([inp['input_ids'] for inp in loader])
        attn_masks = torch.cat([inp['attention_mask'] for inp in loader])

        _ = self.model(input_ids.to(self.device), attn_masks.to(self.device))

        # for inp in loader:
        #     # todo early stop?
        #     _ = self.model(inp['input_ids'].to(self.device),
        #                    inp['attention_mask'].to(self.device)
        #                    )



    def save_data(self, file_name_weights, file_name_inputs, subsets=None):
        self.save_tensors_to_hdf5(self.cached_weights, file_name_weights, subsets)
        self.save_tensors_to_hdf5(self.cached_inputs, file_name_inputs, subsets)

    @staticmethod
    def save_tensors_to_hdf5(data, file_name, subsets=None):
        import h5py
        with h5py.File(file_name, 'w') as f:
            if subsets is None:
                for name, tensor in data.items():
                    f.create_dataset(name, data=tensor.numpy())
            else:
                for name in subsets:
                    if name in data:
                        f.create_dataset(name, data=data[name].numpy())

    def clean_up(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def get_caches(self):
        dst={} 
        for k,v in self.cached_inputs.items():
            v = torch.cat(v, dim=0) 
            v = v.reshape(-1, v.shape[-1])  
            dst[k] = v 
        return dst, self.cached_weights 
    
    def get_XW(self):
        X,W=self.get_caches() 
        X=list(X.values())[0]
        W=list(W.values())[0]
        return X,W 
    
    def clear_caches(self):
        self.cached_weights = {}
        self.cached_inputs = defaultdict(list)
    
    def __del__(self):
        self.clean_up() 
    
    def rehook(self, layers_to_hook):
        if not isinstance(layers_to_hook, list): layers_to_hook = [layers_to_hook]
        self.clean_up()
        self.layers_to_hook = set(layers_to_hook) if layers_to_hook else None
        self.clear_caches()
        self.init_cache()
    
    def set_model_weights(self, layer_name, new_weights):
        """Set the weights of a specific layer by name."""
        for name, layer in self.model.named_modules():
            if name == layer_name and hasattr(layer, 'weight'):
                with torch.no_grad():
                    # Assuming new_weights is a Tensor and already on the correct device and format
                    assert layer.weight.data.shape == new_weights.shape 
                    layer.weight.data = new_weights.to(self.device)
                    # give W, will be used as XW^T
                break
        else:
            raise ValueError(f"Layer with name {layer_name} not found or does not have weights.")

    def get_model_weights(self, layer_name):
        """Get the weights of a specific layer by name."""
        for name, layer in self.model.named_modules():
            if name == layer_name and hasattr(layer, 'weight'):
                return layer.weight.data
        else:
            raise ValueError(f"Layer with name {layer_name} not found or does not have weights.")


  
def get_min_s(weights, n_bits=4): 
    w_min = np.min(weights)
    w_max = np.max(weights)
    num_levels = 2**n_bits - 1
    scale = (w_max - w_min) / num_levels
    return scale, w_min

def quantize(weights, scale=None, w_min=None):
    if scale is None: 
        scale, w_min=get_min_s(weights)
    assert w_min is not None
    quantized_indices = np.round((weights - w_min) / scale).astype(int)
    return quantized_indices, scale, w_min

def quant_dequant(weights): 
    quantized_indices, scale, w_min = quantize(weights) 
    dequantized_weights = w_min + scale * quantized_indices
    return dequantized_weights

import torch.nn as nn
import torch
def quantize_dequantize(X, reduce_dim=-1, bit=8, sym=0): 
    ori_type=X.dtype
    
    if sym: 
        # Calculate the absolute maximum along the specified dimension
        max_val = torch.max(torch.abs(X), dim=reduce_dim, keepdim=True).values
        
        # Calculate the scale factor based on the absolute maximum and number of bits
        scale = max_val / (2**(bit-1) - 1)
        
        # Quantize by scaling and rounding
        quantized_X = torch.round(X / scale)
        
        # Clip the quantized values to fit within the range of an 8-bit signed integer
        quantized_X = torch.clamp(quantized_X, -2**(bit-1), 2**(bit-1) - 1)
        
        # Dequantize by multiplying the quantized values by the scale
        dequantized_X = quantized_X.to(ori_type) * scale
    else: # asym
        # print("!")
        # Calculate the minimum and maximum values along the specified dimension
        min_val = torch.min(X, dim=reduce_dim, keepdim=True).values
        max_val = torch.max(X, dim=reduce_dim, keepdim=True).values
        
        # Calculate the scale and zero_point
        scale = (max_val - min_val) / (2**bit - 1)
        assert not (scale==0).any().item() 
        zero_point = -min_val / scale
        zero_point = torch.round(zero_point).clamp(0, 2**bit - 1)
        # why zero point is rounded to an integer value : 
        #   1. so that the real value of zero is exactly representable. This will result in a slight adjustment to the real representable range [β, α]
        #   2. for int4xint4 gemm kernal (in later-on inference stage)
        
        # Quantize by scaling, adding zero_point, and rounding
        quantized_X = torch.round(X / scale + zero_point) 
        
        # Clip the quantized values to fit within the range of an 8-bit unsigned integer
        quantized_X = torch.clamp(quantized_X, 0, 2**bit - 1)
        
        # Dequantize by multiplying the quantized values by the scale and adjusting by the zero_point
        dequantized_X = (quantized_X.to(ori_type) - zero_point) * scale
        # In practice, will offset first may out of range of int4 ??
    assert not torch.isnan(dequantized_X).any().item()
    return dequantized_X


import torch.nn.functional as F 
class RTNLinear(nn.Module):
    def __init__(self, layer, xbit=8, wbit=8): 
        super(RTNLinear, self).__init__()
        self.weight=layer.weight
        self.bias=layer.bias
        self.xbit=xbit 
        self.wbit=wbit

    def forward(self, x):
        if self.xbit!=16:        
            xp = quantize_dequantize(x, bit=self.xbit) 
        else:
            xp = x.half() 
        # print((xp-x).abs().sum())
        wp = quantize_dequantize(self.weight, bit=self.wbit)  
        return F.linear(xp, wp, self.bias) 

class RTNLinearForGPTQ(nn.Module):
    def __init__(self, layer, xbit=8, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer = layer 
        self.xbit=xbit
        
    def forward(self,x):
        xp = quantize_dequantize(x, bit=self.xbit) 
        return self.layer(xp) 

        
def replace_linear_with_RNT_linear(model, xbit=8, wbit=8 ):
    for name, module in model.named_children():
        if 'lm_head' in name or 'embed' in name: continue # gptq lmquant by default skip this layer
        if isinstance(module, nn.Linear):
            ## Replace the linear layer with a quantized version
            print("replace ", name) 
            setattr(model, name, RTNLinear(module, xbit,wbit))
        elif 'QuantLinear' in str(module.__class__):
            print("replace ", name) 
            setattr(model, name, RTNLinearForGPTQ(module, xbit))
        else:
            # Recursively apply to submodules
            replace_linear_with_RNT_linear(module,xbit,wbit)


