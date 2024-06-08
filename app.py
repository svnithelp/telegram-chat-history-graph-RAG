import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import networkx as nx
import matplotlib.pyplot as plt

# Step 1: Parse the JSON data
with open('result.json') as f:
    data = json.load(f)

# Step 2: Construct the Graph
G = nx.Graph()
for message in data['messages']:
    G.add_node(message['from'], type='user')
    G.add_node(message['id'], type='message')
    G.add_edge(message['from'], message['id'])

# Step 3: Generate Text with GPT-3
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def generate_text(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example prompt generation
prompt = "What's the latest news?"
response = generate_text(prompt)
print("Response:", response)

# Visualize the graph (optional)
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=1000, node_color='lightblue', font_size=10)
plt.title(data['name'])
plt.show()
