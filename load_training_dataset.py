from datasets import load_dataset
from transformers import AutoTokenizer

if __name__ == '__main__':
    print("Loading training dataset.")
    dataset = load_dataset("kimhammar/CSLE-IncidentResponse-V1", data_files="examples_16_june.json")
    instructions = dataset["train"]["instructions"][0]
    answers = dataset["train"]["answers"][0]
    print("Training dataset loaded successfully.")
