# %%

import torch as t

from nnsight import LanguageModel
from dictionary_learning.dictionary import AutoEncoder
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.training import trainSAE
from dictionary_learning.evaluation import evaluate

from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets import load_dataset

import wandb

# %%

class AERun():
    def __init__(self, model=None, submodule=None, activation_dim=None, 
                 dataset_name=None, lr=None, sparsity_penalty=None, 
                 scale_factor=None, ghost_threshold=None, model_name=None):
        dataset = load_dataset(dataset_name, split='train', streaming=True)
        
        def gen():
            for x in iter(dataset):
                yield x['story']
        
        self.buffer = ActivationBuffer(
            gen(),
            model,
            submodule,
            out_feats=activation_dim, # output dimension of the model component
            n_ctxs=1e3,
        ) 

        self.model = model
        self.submodule = submodule
        self.lr = lr
        self.sparsity_penalty = sparsity_penalty
        self.scale_factor = scale_factor
        self.ghost_threshold = ghost_threshold
        self.activation_dim = activation_dim
        self.dictionary_size = activation_dim * scale_factor
        self.dataset_name = dataset_name
        self.model_name = model_name

    def train(self, steps):
        self.full_name = f"ae-{self.model_name.split('/')[-1]}-{self.dataset_name.split('/')[-1]}-{self.lr}-{self.sparsity_penalty}-{self.ghost_threshold}-{steps}.pt"

        run = wandb.init(
            project="my-awesome-project",
            name=self.full_name,
    
            config={
                "learning_rate": self.lr,
                "architecture": "SAE",
                "dataset": self.dataset_name,
                "epochs": steps,
                "sparsity_penalty": self.sparsity_penalty,
                "model": self.model_name,
                "scale_factor": self.scale_factor,
                "ghost_threshold": self.ghost_threshold,
            }
        )

        self.ae = trainSAE(
            self.buffer,
            self.activation_dim,
            self.dictionary_size,
            lr=self.lr,
            sparsity_penalty=self.sparsity_penalty,
            ghost_threshold=self.ghost_threshold,
            device='cuda:0',
            steps=steps,
            log_steps=50,
            
        )

        t.save(self.ae.state_dict(), self.full_name)
        wandb.save(self.full_name)

        evaluation = evaluate(self.model, self.submodule, self.ae, self.buffer, 
                              device='cuda:0')
                            
        for k in evaluation:
            wandb.summary[k] = evaluation[k]
        
        wandb.summary.update()

        wandb.finish()
        return self.ae

model = LanguageModel(
    "delphi-suite/delphi-llama2-100k",
    device_map = 'cuda:0'
)

submodule = model.model.layers[0]
expansion_size = 4

activation_dim = 48

# %%

run = AERun(model=model, submodule=submodule, activation_dim=activation_dim, 
            dataset_name='delphi-suite/tinystories-v2-clean', lr=3e-4, 
            sparsity_penalty=1e-3, scale_factor=expansion_size, ghost_threshold=128,
            model_name="delphi-suite/delphi-llama2-100k")

run.train(500)

# %%