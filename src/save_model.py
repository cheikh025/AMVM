from transformers import OPTForCausalLM

model_name = "facebook/opt-125m"
model = OPTForCausalLM.from_pretrained(model_name)

model_save_path = "./opt125model"
model.save_pretrained(model_save_path)