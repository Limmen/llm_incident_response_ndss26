from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == '__main__':
    print("Loading the fine-tuned incident response LLM.")
    model = AutoModelForCausalLM.from_pretrained("kimhammar/LLMIncidentResponse")
    tokenizer = AutoTokenizer.from_pretrained("kimhammar/LLMIncidentResponse")
    print("LLM loaded successfully.")
