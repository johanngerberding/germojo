from string import Template
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.25, top_p=0.95, max_tokens=256)

model = "TheBloke/Llama-2-13B-German-Assistant-v4-AWQ"
llm = LLM(model=model, quantization="awq")

template = Template("""$personality\nUser: $prompt\nAsssistant:""")
personality = "Du bist ein hilfreicher Assistent mit dem Namen Mojo. Du bist hilfreich und humorvoll."
question = "User: Wer bist du und wie geht es dir? Assistant:"

outputs = llm.generate(
    prompts=template.substitute(personality=personality, prompt=question),
    sampling_params=sampling_params,
)

print(template.substitute(personality=personality, prompt=question))
print(outputs[0].outputs[0].text)

