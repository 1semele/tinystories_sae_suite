# %%

import torch as t

from nnsight import LanguageModel
from dictionary_learning.dictionary import AutoEncoder
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.training import trainSAE
from dictionary_learning.evaluation import evaluate

from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets import load_dataset

# %%

model = LanguageModel(
    "delphi-suite/delphi-llama2-100k",
    device_map = 'cuda:0'
)
submodule = model.model.layers[0]

expansion_size = 4

activation_dim = 48 # output dimension of the MLP
dictionary_size = expansion_size * activation_dim

# %%
# data much be an iterator that outputs strings

def hf_dataset_to_generator(dataset_name, split='train', streaming=True):
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    
    def gen():
        for x in iter(dataset):
            yield x['story']
    
    return gen()

data_generator = hf_dataset_to_generator("delphi-suite/tinystories-v2-clean", "train")

device = t.device("cuda" if t.cuda.is_available() else "cpu")
# %%
buffer = ActivationBuffer(
    data_generator,
    model,
    submodule,
    out_feats=activation_dim, # output dimension of the model component
    n_ctxs=1e3,
) # buffer will return batches of tensors of dimension = submodule's output dimension
# train the sparse autoencoder (SAE)

ae = trainSAE(
    buffer,
    activation_dim,
    dictionary_size,
    lr=3e-4,
    sparsity_penalty=1e-3,
    ghost_threshold=4e6,
    device='cuda:0',
    steps=5000
)

t.save(ae.state_dict(), "ae1.pt")

# %%

ae = AutoEncoder(activation_dim, dictionary_size).to(device)
ae.load_state_dict(t.load("ae1.pt"))

# %%

buffer = ActivationBuffer(
    data_generator,
    model,
    submodule,
    out_feats=activation_dim, # output dimension of the model component
    n_ctxs=1e3,
    device="cuda:0",
) # buffer will return batches of tensors of dimension = submodule's output dimension

evaluation = evaluate(model, 
    submodule, ae, buffer, 
    device='cuda:0',
    hist_save_path="my_hist.png"
)

# %%

type(ae)
# %%
