import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pyttsx3
import speech_recognition as sr
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
            # Set temperature if do_sample=True
            temperature=temperature if do_sample else None,
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


def speech_to_text():
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Use default microphone as audio source
    with sr.Microphone() as source:
        print("Listening...")

        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source)

        # Capture audio input
        audio = recognizer.listen(source)

    try:
        # Perform speech recognition
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand what you said.")
        return None  # Return None when speech recognition fails
    except sr.RequestError as e:
        print("Speech recognition request failed:", e)
        return None  # Return None when speech recognition fails


def speak(text):
    # Initialize TTS engine
    engine = pyttsx3.init()

    # Set properties (optional)
    engine.setProperty('rate', 150)    # Speed percent (can go over 100)
    engine.setProperty('volume', 0.9)  # Volume 0-1

    # Speak the text
    engine.say(text)
    engine.runAndWait()


def interact_with_chatbot(model, tokenizer, dataset, max_length=1024, temperature=0.3, do_sample=True):
    print("Let's GO! (say 'quit' to exit)")
    while True:
        # Listen for user input
        user_input = speech_to_text()
        if user_input is None:
            continue  # Retry if speech recognition failed

        user_input = user_input.strip()  # Strip whitespace from recognized text

        if user_input.lower() == "quit":
            break

        # Find the appropriate response from the dataset based on the input prompt
        response = None
        for item in dataset:
            if item["request"] == user_input:
                response = item["response"]
                break

        # If no response is found in the dataset, generate one using the model
        if response is None:
            response = generate_text(
                model, tokenizer, dataset, user_input, max_length, temperature, do_sample)

        # Print the generated response
        print("Chatbot:", response)

        # Speak the generated response
        speak(response)


# Load dataset from JSON file
dataset = load_dataset_from_json("dataset.json")

# Interaction with the chatbot
tokenizer_instance = GPT2Tokenizer.from_pretrained("fine_tuned_model")
interact_with_chatbot(model, tokenizer_instance, dataset,
                      temperature=0.3, do_sample=True)
