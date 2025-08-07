import torch
import time

print("Loading inputs")
inputs = torch.load("cached_inputs.pt")
print("Loaded inputs")
time.sleep(1)

# print(inputs)
print(inputs.keys())
time.sleep(1)
DBname = list(inputs.keys())[0]
print(inputs[DBname])