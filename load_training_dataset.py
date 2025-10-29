from datasets import load_dataset
from transformers import AutoTokenizer

if __name__ == '__main__':
    print("Loading training dataset.")
    dataset = load_dataset("kimhammar/CSLE-IncidentResponse-V1", data_files="examples_16_june.json")
    instructions = list(dataset["train"]["instructions"][0])
    answers = list(dataset["train"]["answers"][0])
    print(f"Training dataset loaded successfully. Number of instructions: {len(instructions)}. "
          f"Number of answers: {len(answers)}.")
