import json
import torch
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import warnings

warnings.filterwarnings("ignore")

# Load the fine-tuned model for inference
tokenizer = GPT2Tokenizer.from_pretrained("fine_tuned_model")
model = GPT2LMHeadModel.from_pretrained("fine_tuned_model")



def generate_text(model, tokenizer, dataset, seed_text, max_length, temperature, do_sample=True):
    model.eval()
    with torch.no_grad():
        # Tokenize the seed text
        input_ids = tokenizer.encode(seed_text, return_tensors='pt')

        # Generate attention mask
        attention_mask = torch.ones_like(input_ids)

        # Generate new tokens
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,  # Pass attention mask
            max_length=max_length,
            num_return_sequences=1,
            temperature=temperature if do_sample else None,  # Set temperature if do_sample=True
            num_beams=1,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id  # Explicitly set pad_token_id
        )

        # Decode the generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        return generated_text



def load_dataset_from_json(file_path):
    with open(file_path, 'r') as f:
        dataset = json.load(f)
    return dataset





def interact_with_chatbot(model, tokenizer, dataset, max_length=1024, temperature=0.3, do_sample=True):
    print("Let's GO! (type 'quit' to exit)")
    while True:
        # Prompt user for input
        user_input = input("You: ")
        if user_input == "quit":
            break

        # Find the appropriate response from the dataset based on the input prompt
        response = None
        for item in dataset:
            if item["request"] == user_input:
                response = item["response"]
                break

        # If no response is found in the dataset, generate one using the model
        if response is None:
            response = generate_text(model, tokenizer, dataset, user_input, max_length, temperature, do_sample)

        # Print the generated response
        print("Chatbot:", response)


# Load dataset from JSON file
dataset = load_dataset_from_json("dataset.json")


# Interaction with the chatbot
tokenizer_instance = GPT2Tokenizer.from_pretrained("fine_tuned_model")
interact_with_chatbot(model, tokenizer_instance, dataset, temperature=0.3, do_sample=True)

