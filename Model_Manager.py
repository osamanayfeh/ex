from numpy import dtype

from hardware_prepartion import torch, gpu_preparation
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from hugging_face_authentication import hugging_face_auth

class ModelManager():
    MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
    
    def __init__(self, model_id= MODEL_ID, quantization_setting= True):
        if gpu_preparation():
            hugging_face_auth()
            self.model_id = model_id
            self.tokenizer = self.load_tokenizer()
            self.model = self.load_model(self.model_id, quantization_setting)
        else:
            raise EnvironmentError("No GPU detected. This Model requires a CUDA-compatible GPU to run.")
               
    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

    def quantization_setting(self):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
    def load_model(self, model_id, quantization_setting):
        kwargs = {
            "device_map" : "auto",
            "torch_dtype" : "auto",
            "trust_remote_code" : True
        }
        
        if quantization_setting:
            kwargs["quantization_config"] = self.quantization_setting()            
        return AutoModelForCausalLM.from_pretrained(
            model_id, **kwargs
        )
    
    
    def __repr__(self):
        return(
            f"model_manager:\n"
            f"model_id = {self.model_id}\n"
            f"quantization = {self.tokenizer}\n"
            f"model = {self.model}\n"
            f"device = {torch.cuda.get_device_name(0)}"
        )
if __name__ == "__main__":
    print("---------------------------------------------------")
    X = ModelManager()
    print(X)