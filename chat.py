import os 
from dotenv import load_dotenv
import torch 
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM 

load_dotenv(".env")
device = "cuda"
token = os.environ["HUB_TOKEN"] 

system_prompt = """Dies ist eine Unterhaltung zwischen einem intelligenten, hilfsbereitem KI-Assistenten und einem Nutzer.
Der Assistent gibt ausführliche, hilfreiche und ehrliche Antworten."""

prompt_format = "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
prompt = "Wie findest du Atomenergie?"


generator = pipeline(
    model="LeoLM/leo-mistral-hessianai-7b-chat", 
    device=device, 
    torch_dtype=torch.float16, 
    token=token,
)


def main(): 
    out = generator(
        prompt_format.format(system_prompt=system_prompt, prompt=prompt), 
        do_sample=True, 
        top_p=0.95, 
        max_length=8192,
    )
    print(f"User: {prompt}")
    out = out[0]['generated_text']
    search = "<|im_start|>assistant\n"
    out = out[out.index(search) + len(search):]
    print(f"Assistant: {out}")




if __name__ == "__main__": 
    main()