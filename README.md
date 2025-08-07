# Table Of Contents
- [File Structure](#file-structure)
- [Setup](#setup)
- [Running/Usage](#runningusage)
  - [Settings](#settings)
  - [Running with GPTQ starting weights](#running-with-gptq-starting-weights)
  - [Running with SqueezeLLM starting weights](#running-with-squeezellm-starting-weights)
  - [Algorithm Parameters](#algorithm-parameters)
- [Module Explanations](#module-explanations)
    - [Debugging/Testing Scripts](#debuggingtesting-scripts)
        - [perplexity_eval.py](#perplexity_evalpy)
        - [1.1-gptq-perplexity.py](#11-gptq-perplexitypy)
        - [read_analyze_data.py](#read_analyze_datapy)
        - [analyze_data.ipynb](#analyze_dataipynb)
    - [Main Running Scripts](#main-running-scripts)
        - [quantize_model.py](#quantize_modelpy)
        - [full_layer.py](#full_layerpy)
        - [Config.py](#configpy)
    - [Main Algorithm Scripts](#main-algorithm-scripts)
        - [ALNS.py](#alnspy)
        - [remove_operators.py](#remove_operatorspy)
        - [repair_operators.py](#repair_operatorspy)
        - [State.py](#statepy)
        - [Solution.py](#solutionpy)
        - [utils.py - See the SqueezeLLM functions here](#utilspy)
- [Running SqueezeLLM for opt-125m model](#running-squeezellm-for-opt-125m-model)
- [Tomography](#tomography)
- [FIR Design](#fir-design)
- Important Notes:
  - If using `SqueezeLLM` weights (and the quantization grid is non-uniform), then in `local_search`, `delta_q` must be `1`.
  - Using 
  - `RoundToNearest` functions only work for uniform quantization grid.
  - Even with `debug` mode off, it still produces quite a lot of output. I would suggest always redirecting output to a file:
    - `python3 src/GPU/quantize_model.py > [filename].txt`
  - When saving the model (specified by `save_weights=True` and `[stored_weights_path]`), make sure you are saving to a new `.h5` file, otherwise it won't save to existing file.
  - There isn't a flag to turn off L2 norm hill-climbing optimization, but I marked where I used L2 norm in the comment `#L2NORM`.
    - There are only 2 files that use this: `State.py` and `repair_operators.py`, so check those


### File Structure
Before running anything, make sure you have the following project structure. You may need to create the folders if they do not exist.
```
or4ai_quantization
|-- Results
|   |-- full_data
|-- full_data
|   |-- cached_weights.h5
|   |-- cached_weights-transposed.h5
|   |-- cached_inputs_part1.h5
|   |-- cached_inputs_part2.h5
|-- full_data_output
|   |-- tmp
|-- src
|   |-- GPU
|   |   |-- ALNS
|   |   |-- SqueezeLLM
|   |-- SqueezeLLM-gradients
```

- Make sure you have the `full_data_output/tmp` folder created. This is where the algorithm stores the result of each row before combining them at the end into a matrix.
- The `.h5` files in `full_data` aren't necessary unless you are running `read_analyze_data.py` or `analyze_data.ipynb`.

- L_inf/L2 norm data saved by testing scripts will appear in `Results/full_data` (by default, you can change this though)
- saved .h5 weights should be in `full_data_output`

## Setup
Install the necessary Python packages in ```src/requirements.txt```.
```
pip install -r src/requirements.txt
```
Note: `torch` needs to be installed with CUDA support. I used `torch2.3.1` and CUDA version `12.2`

## Running/Usage

- The script to quantize a model is at ```src/GPU/quantize_model.py```.
- Before running, make sure you have the required [file structure](#file-structure).


### Settings
1. Before running, go to ```src/GPU/full_layer.py```, and specifically the ```get_config_settings``` function.
Modify the parameters returned in the ```Config```class.

`Config` class parameters: (only the tuneable parameters - there are other constant parameters)
```
num_gpu: number of GPUs used
rows_per_gpu: number of instances per GPU
index_iterations: number of rows to quantize (for debugging, usually it is all, i.e. weights.shape[0])
use_multiprocess: flag to use multiprocessing (for debugging purposes, should be on by default)

seconds: seconds to run per row
nQuantized: number of bits

save_weights: save weights to a file
stored_weights_path: path to where we store the weights (stores as a .h5 file)
save_to_file: a bool flag, save L_inf per row to a file (optional, for analysis purposes)
- save_to_file might have some unexpected errors, don't use it if you are quantizing a model. 
- Only use it for testing
output_filename: path to where we store the L_inf information

nQuantized: number of bits

use_gptq: flag to use gptq_matrix as starting weights
debug: debug flag during ALNS
debug_process: prints debug info for code other than ALNS (process info, etc.)

use_squeezellm: flag to use squeezellm as starting weights
squeezellm_LUT: lookup table associated with this matrix

keep_outliers: flag to keep outliers or not
outlier_range: the magnitude for a weight to be considered an outlier
```

- Important: For `rows_per_gpu`, it is different for the `fc2` matrix. To change number of rows per GPU used, go to `full_layer.py, quantize_given_matrix()` and change this block of code:
  - ```python
    if 'fc2' in DBname:
        ROWS_PER_GPU = 5
    else:
        ROWS_PER_GPU = 10
    ``` 
- If using `save_to_file`, then the path `output_filename` saves the results to `Results/full_data/output_{OUTPUT_FILENAME}`

 
For the `opt-125m` model, an `outlier_range` of `0.2496` gives `0.45%` outliers. For any other range, use [analyze_data.ipynb](#analyze_dataipynb) to determine the ranges for specific percentage of outliers

2. (Optional) If you want to use CPU only when GPUs are available, set `num_gpu` to 1 and in `full_layer.py`, at the beginning of the function `executeALNS`, comment out these two lines:

```    
if torch.cuda.is_available():
    torch.cuda.set_device(device)
```
3. If you want to start from SqueezeLLM or GPTQ weights, go to [GPTQ](#running-with-gptq-starting-weights) or [SqueezeLLM](#running-with-squeezellm-starting-weights)


4. Finally, cd into the base `or4ai_quantization` directory, and run:
```
python3 src/GPU/quantize_model.py
```

Getting the c4/wikitext2 datasets may fail, you may need to run it multiple times if it says it's not in the cache.


### Running with GPTQ starting weights

1. Run ```src/GPU/1.1-gptq-perplexity.py``` to save GPTQ starting weights to `[GPTQ_path]`, follow [1.1-gptq-perplexity.py](#11-gptq-perplexitypy)
2. In `quantize_model.py`, set `gptq_weight_path` to `[GPTQ_path]`
3. In `full_layer.py`, function `get_config_settings`, change `use_gptq` to `True` and `use_squeezellm` to `False`


### Running with SqueezeLLM starting weights

1. Follow [Running SqueezeLLM](#running-squeezellm-for-opt-125m-model) on the model, save the packed model (using `pack.py`) to path `[packed_path]`
2. In `quantize_model.py`, set `squeezellm_state_dict_path` to `[packed_path]`
3. In `full_layer.py`, function `get_config_settings`, change `use_squeezellm` to `True`, and `use_gptq` to `False`.
4. Follow steps shown above in [Running/Usage](#runningusage)


If you want to run it from nearest (this only works for uniform quantization grid), simply set `use_gptq` and `use_squeezellm` to `False` in `full_layer.py, get_config_settings()`.

### Algorithm Parameters
These are adjustable parameters for our algorithm.

- In ```src/GPU/ALNS/operator_utils.py:```
  - `DESTROY_RATE` percentage of the array to change after every destroy/repair pair
  - `ALPHA_COEFFICIENT` (not used until `worst_remove` is implemented): coefficient of `a` for `worst_remove` scoring criteria

- In `src/GPU/ALNS/local_search.py:`
  - `SWAP_EVAL_BATCHES` number of batches we have for `q_1` weights during swap
  - `NUM_FILTERS` size of k_e, used for epsilon filtering (filtering swaps based off small number of D_ks)
  - `EPSILON` new swap has to be better by at least `EPSILON` (prevents infinite loops)
  - `BIAS`  giving a more conservative estimate for swap condition (to prevent infinite loops)



## Module Explanations
Explanation of each file and how to run them.

### Debugging/Testing Scripts

All scripts should be run from the ```or4ai_quantization``` directory.
- ```perplexity_eval.py``` prints out the perplexity of a model given the quantized weights.
- ```1.1-gptq-perplexity.py``` is used to generate the starting GPTQ weights
- ```read_analyze_data.py``` can be used to see L infinity, L2 norm of the quantized matrix
- `analyze_data.ipynb`, is used to calculate generate plots or calculate percentage outliers

#### perplexity_eval.py
- In the `main` block:
- Runs the model `MDL` on the datasets `datasets` (can be `c4` or `wikitext2`), on either the eval or calibration dataset, and print out its perplexity.

- Specify the path of the quantized weights in `quantized_weights_path`. The weights in this path should be saved using `utils.save_tensors_to_hdf5()` function

- `use_calibration_dataset` specifies whether or not you use the calibration or eval dataset
- To save the inputs `X`, uncomment the code in the `dataset` for loop.



#### 1.1-gptq-perplexity.py
- Runs GPTQ and saves the weight matrices to a `dict[str, torch.Tensor]` in the path `weight_save_path`
- Specify the model in the variable `MDL`.
- To change the settings, change variable `quantize_config`.
  - We used these settings: `bits=3`, `group_size=-1`, `sym=False`

#### read_analyze_data.py
- Computes L_inf and L2 norm for quantized weights, used for testing purposes.
- Make sure that the original weights (transposed) are stored in `./full_data/cached_weights-transposed.h5`
- Specify the model in `MDL`, `datasets` as a list of strings
- Uses the `X` at path `input_file_paths`
  - This is saved by using `torch.save` on a dict.
- `quantized_weight_path` being the dequantized values, stored as `numpy` arrays in a `.h5` file
- Saves the data in `CSV_FILEPATH`

- Note: `CALC_BOTH` was for testing, don't use this

- In order to see `||X_qW_q - X_qW||`, you need to get the values of `X_q` first for each matrix.
  - To do this, go to [perplexity_eval.py](#perplexity_evalpy), uncomment the code in the `datasets` for loop, and it will save it to `[input_save_path]`.
  - Then, in `1.1-gptq-perplexity.py`, change the `input_file_paths` to the `[input_save_path]`, and run the script

Important notes:
- If using 16GB GPUs, it may OOM, mainly on the `fc1` matrix. In this case, you might have to set it to run on CPU for these matrices by changing `device` before the `input_file_path` loop. 


#### analyze_data.ipynb

- Use this for quick tests on data. Code for plotting weights is here, as well as checking how many outlier weights are in a given range.
  - Run the cells sequentially. 
  - Percentage outlier checking is in the 2nd cell, just change the `bound` variable.




### Main Running Scripts

Scripts used to run the quantization.

#### quantize_model.py

- This is the main script to quantize the model.w
- Before running, specify the settings in [`full_layer.py`](#full_layerpy)
- In the `main` block, specify the path to the model (e.g. `facebook/opt-125m`)
  - If starting from GPTQ weights:
    - The GPTQ weights at `[gptq_weight_path]` should be stored in a `.h5` file using `utils.save_tensors_to_hdf5`
  - If you want to use SqueezeLLM, specify the path to the `state_dict`. 
    - `state_dict` should point to the output after the (final) `packing` step of SqueezeLLM
- Afterwards, go into the `full_layer` script and change the settings in `get_config_settings`.
- Then, you can run this script to start the quantization. Refer to [Running/Usage](#runningusage).

Logic:

- For each matrix in sequential order, run inference on all `128` (default number) batches and it uses the `DataCache` to store all the inputs to this matrix
- Then, use `process_data` to combine into 1 matrix `Xq` and then call our `quantize_matrix` function using these activations
  - Therefore, we are optimizing objective `||X_qW_q - X_qW||`


#### full_layer.py

- Note: don't run this file directly, it most likely won't work. `full_layer_path` and `get_testing_settings` can be safely removed, they were here for testing purposes
- This is the file to quantize a matrix, through the function `quantize_given_matrix()`
- 
Logic:

- Order of calls:
  - `quantize_model.py` calls `quantize_given_matrix() -> quantize_matrix() -> quantize_matrix_concurrently() -> executeALNS()`
- In `quantize_matrix`, after attempting to quantize a matrix, if there were any rows that failed to quantize (usually because of OOM), it will try again with half the number of GPUs per row.
- Otherwise, it will just use nearest and save that. (Note: This doesn't work for SqueezeLLM since quantization grid is not uniform)

#### Config.py

- A dataclass use only to store settings. Please see the documentation in the class.

### Main Algorithm Scripts

The main quantization algorithm.

#### ALNS.py

- A wrapper class for the main `ALNS` algorithm.
- Initialize the algorithm using the init function and the setter functions.
  - If you want to add/remove remove or repair operators, add it to the `solve()` function
- `solve()` runs the `ALNS` algorithm with the specified settings and returns a `Solution` object storing information about the run.

#### remove_operators.py

- Contains two operators, `random_remove` and `worst_remove`.
- Don't use `worst_remove`, it hasn't been updated to work with our current algorithm
  - Check slides to implement this with GPU support
- `random_remove` marks `DESTROY_RATE %` of weights as destroyed, creates a new `State` class.
  - This is used by the `repair` operators to modify the weights.

#### repair_operators.py

- Contains two operators `random_repair` and `greedy_repair`
- Don't use `greedy_repair`, it hasn't been updated either.
- `random_repair` looks at the weights marked as removed by the `remove operator` and randomly changes it by `+1, -1, or 0`.

#### State.py
- Class representing the main state of the algorithm.
- Stores the current weights, `D_ks`, current objective, etc.

- Most important function is the `objective()`, which returns the L_inf norm of the current state
- For `objective()`, when you make a change, make sure to set `state.recalculate_flag = True` before calling `state.objective`
- To make a change to a weight at index `idx`:
  - modify `weights[idx]` to a new integer, and take note of the change: `delta_idx`
  - Then, append `(idx, delta_idx)` into `state.changes` (this is a list)
  - Once you need the objective again, change `state.recalculate_flag = True` and then call `state.objective()
- To get the quantized weights in FP16, simply call `state.get_quantized_weights()` (no arguments)


#### Solution.py

- A solution class that stores important information during the ALNS run.

#### utils.py

Utility functions, for calculating L_inf, L2 norm, and also functions for dealing with SqueezeLLM.

<ins>SqueezeLLM functions:</ins>

Note: To get the `state_dict`, call `state_dict = torch.load([packed_path])`, where `[packed_path]` is the path to the packed model from SqueezeLLM

- `get_squeezellm_lookup_tables()` takes in the original model, the `state_dict` and returns a dictionary of `[module name] -> lookup_table`
  - so, after this, e.g. `LUT['model.decoder.layers.0.self_attn.k_proj']` will correspond to the lookup table
    - It will have shape `(m, 8)`.
  - By default, it sorts the lookup table as well. 

- `round_to_nearest_pole_sim()`: takes in a weight row `w`, lookup table `poles` and moves it onto torch device `device`.
  - `w` should be floats, and it returns another integer row of the same shape, corresponding to the index in the LUT that has teh nearest corresponding float value




Other unused but potentially useful functions:

- `dequantize_squeezellm_3bit_matrix()` takes in a packed matrix (value corresponding to key `[module_name].qweights` in the `state_dict` after you pack the matrix) and returns the dequantized (integer) values of a matrix
  - e.g. `qweights = state_dict['model.decoder.layers.0.self_attn.k_proj.qweight']`
  - You may need to run `setup.py` in SqueezeLLM and uncomment `import quant_cuda` in `quant.py` (I had some issues with this)
  - Currently, this function is unused since we only need the lookup table and then round to nearest.
- `get_dequantized_squeezellm_model()` takes in the `state_dict` after packing in SqueezeLLM and returns the model with unquantized matrices.





## Running SqueezeLLM for opt-125m model
- In `src/GPU/SqueezeLLM`, the following changes should already be applied.
- Otherwise, if you clone the directory from GitHub, then you need to change the following things for it to work on `opt-125m`:
  - In `quantization/pack.py`, change `sequential_lut = ["q", "k", "v", "o", "up", "down"]`,
  - ```
    sequential_lut_real_name = {
              "q": "self_attn.q_proj",
              "k": "self_attn.k_proj",
              "v": "self_attn.v_proj",
              "o": "self_attn.out_proj",
              "up": "fc1",
              "down": "fc2",
          }
    ```
  - and `layers = model.model.decoder.layers`.
- Then follow the steps outlined in the `README` for SqueezeLLM at `src/GPU/SqueezeLLM`.
  - Make sure to specify `--model-type opt` when running the commands
- After all the steps, the `state_dict` of the resulting quantized model will be at `[PACKED_CKPT_PATH]`.

## Tomography

To run the Tomography design experiment:

1. **Unzip sample images** into your chosen image folder:
   ```bash
   unzip 4levle_images.zip  
   unzip bimnary_images.zip  
    ```

2. **Configure settings** in `tomography.py`:

   * `image_folder`: path to input images (e.g. `./images/`)
   * `grey_levels`: bits per pixel (e.g. 1 = binary, 2 = 4-level)
   * `num_angles`: number of projection angles
   * `num_detectors`: detectors per projection
   * `noise`: uniform noise level
3. **Run the reconstruction**:

   ```bash
   python src/GPU/tomography.py
   ```

## FIR Design


To run the FIR design experiment:
1. **Define parameters** in `fir_config.py` under `project_specs`:
   - `order`: filter order (number of taps − 1)  
   - `p_bits`: coefficient quantization bits  
   - `pass_edge`: normalized first passband edge (0 < f < fs/2)  
   - `stop_edge`: normalized first stopband edge  
   - `pass_edge2`: normalized second passband edge (for bandpass/bandstop)  
   - `stop_edge2`: normalized second stopband edge (for bandpass/bandstop)  
   - `fs`: sampling frequency  
   - `K`: overall gain factor  
   - `grid_density`: frequency‐grid density for the design algorithm  
   - `filter_type`: one of `"lowpass"`, `"highpass"`, `"bandpass"`, `"notch"`, or `"bandstop"`  

2. **Run the standard FIR design**:
   ```bash
   python src/GPU/fir_design.py
     ```

3. **Run the Gurobi‐optimized design**:

   ```bash
   python src/GPU/gurobi_fir.py
   ```


