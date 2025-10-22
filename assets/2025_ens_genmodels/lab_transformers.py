
# %%
########################################################################################
""" Part 1: Intro """

# pip install transformers
# pip install accelerate

# TODO:
# What can you tell about Hugging Face's transformers library ? Why is it useful ? (In a few sentences)
# What can you tell about GPT2 ? (In a few sentences)
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import torch
import numpy as np
import torch.nn as nn
import math

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
torch.set_grad_enabled(False)

# Implementing and training your own tokenizer can be painful
# In this lab, we will use GPT2's pre-trained tokenizer
# You can check out https://github.com/Perceptronium/Tiny-Tokenizer for an implementation from scratch of a basic BPE tokenizer
model_name = 'gpt2'
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
gpt2 = GPT2LMHeadModel.from_pretrained(model_name,
                                       device_map="auto",
                                       pad_token_id=tokenizer.eos_token_id)


# TODO: (Hint: gpt2.device contains information on the location of the model)
# What device are you using ? What can you comment on that ?
raise NotImplementedError

# TODO: (Hint: gpt2.config contains information about the model)
# How many layers does the GPT2 model have ?
# How many heads ?
# How big is the pre-trained vocab ? How does it compare to modern LLMs vocab size ? (look it up)
raise NotImplementedError

# %%
########################################################################################
""" Part 2: Embeddings """
# TODO:
# What is the embedding dimension of GPT2 ?
raise NotImplementedError


def visualise_embeddings(prompt: str, arrows: bool = False):
    input = tokenizer(prompt, return_tensors="pt").to(gpt2.device)

    print(f'{"|token|":8} -> id\n')
    for token in input['input_ids'][0]:
        print(f'{"|" + tokenizer.decode(token)+"|":8} -> {token}')

    # Get the (trained) embedding matrix
    all_embeddings = gpt2.transformer.wte.weight
    # TODO: what shape is this matrix ? (print it) Why ?
    raise NotImplementedError

    # TODO: (Hint input['input_ids'][0] contains the vocab IDs of the tokens in your prompt)
    # Select lines corresponding to the tokens in your prompt
    raise NotImplementedError
    # embeddings = ...

    # Load everything back to CPU
    embeddings = embeddings.cpu()

    # TODO: explain how the following code proceeds to visualize embeddings
    pca = PCA(n_components=2, svd_solver='covariance_eigh', random_state=int(0))
    coordinates = pca.fit_transform(embeddings)
    names = prompt.split()
    plt.close('all')
    _, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(coordinates[:, 0], coordinates[:, 1])
    for (x, y), label in zip(coordinates, names):
        ax.annotate(label, (x, y), xytext=(3, 3), textcoords='offset points',
                    fontsize=9, ha='left', va='bottom')
    if arrows:
        pairs = [(2*i, 2*i + 1) for i in range(embeddings.shape[0] // 2)]
        for ci, ti in pairs:
            dx, dy = coordinates[ti] - coordinates[ci]
            ax.arrow(coordinates[ci, 0], coordinates[ci, 1], dx, dy,
                     width=0.05,
                     head_width=0.1,
                     head_length=0.1,
                     length_includes_head=True, alpha=0.8)

    ax.set_title("GPT-2 token embeddings (PCA visualisation)")
    ax.set_xlabel("Coord. 1")
    ax.set_ylabel("Coord. 2")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# %%
# TODO:
# Visualize the embeddings of the tokens in this prompt, without arrows. What can you comment ?
prompt_1 = " duck goose tiger lion bear train truck plane car boat ship"
raise NotImplementedError

# %%
# TODO:
# Visualize the embeddings of the tokens in this prompt, with arrows. What can you comment ?
prompt_2 = " France Paris Germany Berlin Italy Rome Spain Madrid Belgium Brussels"
raise NotImplementedError

# %%
########################################################################################
""" Part 3: How does a Transformer predict tokens ? """

prompt = 'Hello I am a'
input = tokenizer(prompt, return_tensors="pt").to(gpt2.device)
output = gpt2.generate(input_ids=input['input_ids'],
                       attention_mask=input['attention_mask'],
                       max_new_tokens=1)

# TODO:
# Using the tokenizer, decode the output using the decode method
raise NotImplementedError

# TODO:
# Provide 2 ways to generate more tokens
raise NotImplementedError

# %%
# The model actually outputs a probability vector over all possible tokens
# We can get those with return_dict_in_generate=True, output_scores=True
output = gpt2.generate(input_ids=input['input_ids'],
                       attention_mask=input['attention_mask'],
                       max_new_tokens=1,
                       return_dict_in_generate=True,
                       output_scores=True)

# TODO: (Hint: you can get the scores using output['scores'][0])
# What is the shape of the scores ? Why ?
# Are these probabilities ? Why / Why not ?
# If not, how can you convert them to a probability distribution ? (Hint: torch.nn.functional contains a function that performs this)
raise NotImplementedError

# Given a distribution over the vocabulary, the most obvious way to predict the next token is to choose the most probable one
# This is called the Greedy Search
# TODO:
# predict the next token using a Greedy Search
raise NotImplementedError

# TODO:
# what are alternative ways to predict the next token ?
# How would it impact the model ?
# %%
########################################################################################
""" Part 4: Building blocks of Transformers """

# In this section, you will implement the core building blocks of a decoder-only Transformer using PyTorch

# TODO
# Implement the Feed Forward sub-layer


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff, bias=False)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        # x: [batch_size, seq_len, d_model]

        raise NotImplementedError

# %%
# TODO
# Implement the scaled dot-product attention


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self,
                Q: torch.Tensor,
                K: torch.Tensor,
                V: torch.Tensor,
                causal_mask: torch.Tensor = None):
        """ Compute the Dot Product Attention : Att_head_i = softmax(Q_i@K_i.T / d_head)@V_i """

        # Q: [batch_size, seq_len, d_head]
        # K: [batch_size, seq_len, d_head]
        # V: [batch_size, seq_len, d_head]
        # causal_mask: [seq_len, seq_len]

        d_head = ...

        # TODO
        # Hint: you may use torch.matmul for properly handling the batch dimension

        if causal_mask is not None:
            # TODO (Hint: you may use the masked_fill_ method of torch.Tensor)
            # Mask-out the entries of scores that correspond to entries equal to False
            # in the causal mask by -torch.inf
            raise NotImplementedError

        raise NotImplementedError


seed = 0
batch_size = 2
seq_len = 3
d_head = 4
torch.manual_seed(seed)
Q = torch.randn(batch_size, seq_len, d_head)
K = torch.randn(batch_size, seq_len, d_head)
V = torch.randn(batch_size, seq_len, d_head)
causal_mask = torch.tril(torch.ones(seq_len, seq_len))

attention = ScaledDotProductAttention()
scores = attention(Q, K, V, causal_mask)
print(scores)
# Expected output:
# [[[-9.3348e-02,  6.8705e-01, -8.3832e-01,  8.9182e-04],
# [ 3.1338e-01,  2.1429e-01, -2.1690e-02,  1.5626e-01],
# [ 3.0827e-01,  4.4860e-01,  2.5764e-01,  2.9050e-01]],

# [[-8.0249e-01, -1.2952e+00, -7.5018e-01, -1.3120e+00],
# [-6.8862e-01, -1.1866e+00, -7.1391e-01, -1.1376e+00],
# [ 1.2224e-01, -4.6247e-01, -4.7088e-01, -1.7241e-01]]]

# %%
# TODO
# Implement the causal multihead self-attention sub-layer


class CausalMultiheadSelfAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 device: torch.device = None):
        super().__init__()

        self.n_heads = n_heads

        self.dim_heads = int(d_model // self.n_heads)

        # Linear transforms
        self.w_q = nn.Linear(in_features=d_model,
                             out_features=self.n_heads*self.dim_heads,
                             bias=False,
                             device=device)
        self.w_k = nn.Linear(in_features=d_model,
                             out_features=self.n_heads*self.dim_heads,
                             bias=False,
                             device=device)
        self.w_v = nn.Linear(in_features=d_model,
                             out_features=self.n_heads*self.dim_heads,
                             bias=False,
                             device=device)
        self.w_o = nn.Linear(in_features=self.n_heads *
                             self.dim_heads, out_features=d_model,
                             bias=False,
                             device=device)

        self.compute_attention = ScaledDotProductAttention()

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None):
        """ Implement a causal multihead self attention module
            The main challenge stems from properly handling the tensors' shapes """

        # x shape [batch_size,  seq_len, d_model]
        seq_len = ...

        # Hint 1: Form Queries, Keys and Values for all heads

        # Hint 2: Take a look at the __init__ method

        # Hint 3: You can separate the head dimension and transpose the tensors

        # Create causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))

        # Compute each head's attention with causal masking
        head_attentions = ...

        # Hint 4: After computing each head's outputs, concatenate them and perform the final linear transformation

        raise NotImplementedError


seed = 0
batch_size = 2
seq_len = 3
d_model = 16
n_heads = 4

torch.manual_seed(seed)
x = torch.randn(batch_size, seq_len, d_model)
token_positions = torch.arange(seq_len)

MHA = CausalMultiheadSelfAttention(d_model=d_model, n_heads=n_heads)
output = MHA(x, token_positions)
print(f"MHA output : {output}")

# Expected output:
# MHA output : tensor([[[ 0.4311,  0.1942,  0.0750, -0.7202,  0.5516,  0.3183, -0.0102,
#           -0.2272,  0.0042, -0.2421,  0.6288,  0.1895,  0.3255,  0.3524,
#            0.2454, -0.4726],
#          [ 0.3842, -0.1425, -0.1424, -0.3356,  0.3557,  0.5002, -0.3549,
#            0.2474,  0.2586, -0.2110,  0.3670, -0.1092,  0.2557,  0.4153,
#            0.2628, -0.2379],
#          [ 0.2285, -0.0782, -0.2490, -0.2906,  0.0135,  0.4204, -0.6241,
#            0.3644,  0.3407, -0.1292,  0.2700, -0.3671,  0.0054,  0.1506,
#            0.3263, -0.2177]],

#         [[ 0.1125,  0.0779, -0.2409,  0.0813, -0.2801, -0.2356, -0.2244,
#            0.3209,  0.0731, -0.2492, -0.0603,  0.0865,  0.0234, -0.1469,
#           -0.1721,  0.0960],
#          [ 0.0310,  0.1962,  0.2307,  0.0626,  0.1111, -0.3962,  0.3060,
#           -0.0208,  0.1173, -0.1208,  0.1348,  0.4008,  0.1169,  0.0118,
#           -0.2727,  0.1718],
#          [ 0.0271,  0.1162, -0.1361,  0.1078, -0.2687, -0.2343,  0.0915,
#            0.0874,  0.0268, -0.1082, -0.1162,  0.1609, -0.0597, -0.1441,
#           -0.1793,  0.1804]]])
