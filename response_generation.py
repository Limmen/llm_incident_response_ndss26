import random
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading

if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained("kimhammar/LLMIncidentResponse")
    tokenizer = AutoTokenizer.from_pretrained("kimhammar/LLMIncidentResponse")
    dataset = load_dataset("kimhammar/CSLE-IncidentResponse-V1", data_files="examples_16_june.json")
    instructions = dataset["train"]["instructions"][0]
    answers = dataset["train"]["answers"][0]
    model.eval()
    instruction = random.choice(instructions)
    inputs = tokenizer(instruction, return_tensors="pt").to(model.device)
    gen_kwargs = dict(max_new_tokens=6000, temperature=0.8, do_sample=True)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
    thread = threading.Thread(target=model.generate, kwargs={**inputs, **gen_kwargs, "streamer": streamer})
    thread.start()
    for new_text in streamer:
        print(new_text, end="", flush=True)
