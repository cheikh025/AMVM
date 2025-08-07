import gc

import os
import torch
from torch import nn
import utils.utils as ut

from full_layer import quantize_given_matrix


class QuantizedLinear(nn.Module):
    """
    Unused QuantizedLinear class
    To use, replace Linear layers in the model with this class, and just perform inference
    This class will quantize the matrix with the inputs during inference.

    However, this only works if entire input is passed through (cannot process in batches)
    Currently, there's out of memory errors if we use this with full input.

    """
    def __init__(self, original_linear_layer: torch.nn.Linear) -> None:
        super(QuantizedLinear, self).__init__()
        self.original_linear = original_linear_layer
        self.has_quantized = False

        # self.gptq_weight_path = None
        self.gptq_weight_path = "gptq_asym_weights_trick.h5"

    def quantize_weights(self, inputs: torch.tensor) -> None:
        if self.has_quantized:
            return  # Already quantized
        self.has_quantized = True
        # Otherwise, call the main function to quantize the weights
        weights = self.original_linear.weight.detach().cpu().numpy()

        print(f"Weights has dimension {weights.shape}")
        print(f"Inputs has dimension {inputs.shape}")
        if inputs.dim() == 3:
            inputs = torch.flatten(inputs, 0, 1)

        print(f"Inputs after flatten has dimension {inputs.shape}")
        inputs_numpy = inputs.detach().cpu().numpy()
        self.original_linear.weight.data = torch.tensor(quantize_given_matrix(self.path_name, inputs_numpy, weights, self.gptq_weight_path), dtype=torch.float16,
                                                        device=self.original_linear.weight.device)

    def forward(self, inputs: torch.tensor):
        if not self.has_quantized:
            gc.collect()
            torch.cuda.empty_cache()
            self.quantize_weights(inputs)

        # self.original_linear.weights.data = self.quantized_weights
        res = self.original_linear(inputs)
        del inputs
        gc.collect()
        torch.cuda.empty_cache()
        return res

def replace_with_quantized(model) -> None:
    """Replaces all the Linear layers in model with QuantizedLinear classes"""
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_with_quantized(module)

        if isinstance(module, nn.Linear):
            quantized_linear = QuantizedLinear(module)
            setattr(model, name, quantized_linear)

def add_name_to_modules(model) -> None:
    for name, module in model.named_modules():
        module.path_name = name

import sys
import os
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from utils.modelutils import TextGenerator
from utils.netutils import no_ssl_verification
from utils.datautils import get_loaders
from utils.modelutils import LayerDataCache
import torch
import time
import h5py


def get_linear_layer_names(model) -> list[str]:
    """
    Gets the list of strings corresponding to linear layers, in execution order
    """
    DBnames = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            DBnames.append(name)
    return DBnames

class StopException(Exception):
    """Custom exception in order to stop the model after saving activations for a specific layer"""
    pass

class DataCache:
    """
    Class to store inputs going through the model in each batch
    Fields:
        - data_cache: list of torch Tensors, storing inputs to a module for each input batch
    """
    # List of tensors, corresponding to each batch's X value
    data_cache: list[torch.Tensor]
    def __init__(self):
        self.data_cache = []

    def get_hook(self):
        """
        Returns a callable hook which saves the inputs to a specific module in data_cache
        """
        def save_weights_hook(module: nn.Module, input: torch.Tensor, output) -> None:
            # Save the inputs
            inputs = input[0].detach().cpu()
            self.data_cache.append(inputs)
            # Raise a custom exception to stop the program
            raise StopException

        return save_weights_hook

    def attach_hook(self, model, name: str) -> torch.utils.hooks.RemovableHandle:
        """Attaches this hook to the module with name 'name'
        """
        for layer_name, module in model.named_modules():
            if name == layer_name:
                return module.register_forward_hook(self.get_hook())
        return None


def process_data(data_list: list[torch.Tensor], DBname: str) -> torch.Tensor:
    """Takes in the data_list from data_cache and combines them into a matrix of size k x m
    """

    ret = torch.concat(data_list, dim=0)
    # For our opt125 model:
    # FC matrices inputs in data_cache shape [128, 2048, (768 or 3072)] for fc1/fc2 respectively after concat
    # Other matrices have shape [262144, 768]
    if 'fc' not in DBname:
        ret = ret.flatten(0, 1)
    return ret


def quantize_and_replace_matrix(model, DBname: str, train_loader, gptq_weight_path: str=None,
                                squeezellm_state_dict: dict[str, torch.Tensor]=None) -> None:
    """
    Runs inference on all calibration data (train_loader)
    Quantizes the module with name DBname using this calibration data, then replaces the module in the model with
    this new quantized module
    """

    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")


    print(f"On {DBname}")
    timeStart = time.time()
    data_cache = DataCache()
    # Attach the hook to layer DBname
    handle = data_cache.attach_hook(model, DBname)

    # Run inference for every batch in our calibration set
    # The data cache will store the input to
    with torch.no_grad():
        iteration = 0
        for inp in train_loader:
            try:
                _ = model(inp['input_ids'].to(device),
                          inp['attention_mask'].to(device)
                          )
            except StopException:
                # Early stop here once we save the weights
                pass
            print(f"Finished batch iteration {iteration}")
            iteration += 1

    # Remove handle
    handle.remove()
    print(f"Total time taken to perform inference and collect data for {DBname} is {time.time() - timeStart} sec")

    # Create input matrix from the list of inputs
    inputs = process_data(data_cache.data_cache, DBname).numpy()
    print(f"Inputs has shape {inputs.shape}")

    squeezellm_LUT_dict = None
    if squeezellm_state_dict is not None:
        print("Getting SqueezeLLM lookup table")
        squeezellm_LUT_dict = ut.get_squeezellm_lookup_tables(model, squeezellm_state_dict)

    weights = None
    # Get weights of module with name DBname
    for layer_name, module in model.named_modules():
        if DBname == layer_name:
            weights = module.weight.data.detach().cpu().numpy()
            break

    # Quantize the matrix, then replace weights for this module in the model with this new quantized matrix
    print("Starting quantization")

    squeezellm_LUT = None
    if squeezellm_LUT_dict is not None:
        # Get the lookup table for this specific matrix
        squeezellm_LUT = squeezellm_LUT_dict[DBname]

    quantized_weights = quantize_given_matrix(DBname, inputs, weights, gptq_weight_path, squeezellm_LUT)

    for layer_name, module in model.named_modules():
        if DBname == layer_name:
            module.weight.data = torch.tensor(quantized_weights, dtype=module.weight.data.dtype).to(
                module.weight.data.device)
            break


@torch.no_grad()
@torch.inference_mode()
def main(DS: str, model_path: str, gptq_weight_path: str, squeezellm_state_dict_path: str=None) -> None:
    """
    Main function to quantize a model at model_path
    Using calibration dataset DS, and starting weights at gptq_weight_path.
    If using squeezeLLM, provide the path to the state_dict
    """
    # Environment variables for debugging
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"
    # Read in the model
    nSentence, tokenPerSentence = 128, 2048  # 128

    with no_ssl_verification():
        opt_generator = TextGenerator(model_path)
        model = opt_generator.model.half()
    print("Done reading in model")


    # Read the datasets
    with no_ssl_verification():
        train_loader, val_loader = get_loaders(DS, model=model_path, nsamples=nSentence)
        # TODO: cache data locally to prevent rereading every time

    print("Loaded data")
    print("Starting...")
    linear_layer_names = get_linear_layer_names(model)

    # Read in the state dict
    if squeezellm_state_dict_path is not None:
        squeezellm_state_dict = torch.load(squeezellm_state_dict_path)
    else:
        squeezellm_state_dict = None

    for DBname in linear_layer_names:
        quantize_and_replace_matrix(model, DBname, train_loader, gptq_weight_path, squeezellm_state_dict)


if __name__ == '__main__':
    DS = "c4"  # Dataset name
    # gptq_weight_path = "gptq_asym_weights_trick.h5"  # Weight for starting GPTQ weight
    gptq_weight_path = None
    model_path = "facebook/opt-125m"
    squeezellm_state_dict_path = None#"./src/GPU/SqueezeLLM/dense_only_packed"
    main(DS, model_path, gptq_weight_path, squeezellm_state_dict_path)



