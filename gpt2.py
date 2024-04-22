import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load data from JSON file
with open('dataset.json', 'r') as f:
    data = json.load(f)

# Extract requests and responses from the dataset
requests = [sample['request'] for sample in data]
responses = [sample['response'] for sample in data]

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Tokenize input sequences
inputs = tokenizer(responses, return_tensors='pt', truncation=True, padding=True)

# Fine-tune the model
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data["input_ids"])

    def __getitem__(self, idx):
        return {key: tensor[idx] for key, tensor in self.tokenized_data.items()}

# Create custom dataset
dataset = CustomDataset(inputs)

# Define training parameters
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
num_epochs = 3

# Fine-tune the model
for epoch in range(num_epochs):
    for batch in torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True):
        optimizer.zero_grad()
        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Save fine-tuned model
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")

# Load the fine-tuned model for inference
tokenizer = GPT2Tokenizer.from_pretrained("fine_tuned_model")
model = GPT2LMHeadModel.from_pretrained("fine_tuned_model")

# Generate response to the question "What is a CPU?"
question = "What is a CPU?"
input_ids = tokenizer.encode(question, return_tensors='pt')

# Generate response
output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

# Decode and print the response
response = tokenizer.decode(output[0], skip_special_tokens=True)
print("Response:", response)
