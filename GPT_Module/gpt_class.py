import torch 
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class FrozenGPT2:
    def __init__(self,model_name: str = "gpt2", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval() #setting the model to eval mode since we don't train it 
        self.model.requires_grad_(False) #Since we are not training no need to update grads 
        self.model.to(self.device) # Moves the model to the device we are on


    # Breaking down the fucntion:
        # text: Str is just a type hint we see in other functions
        # -> torch.Tensor, this is also type saftey and says this function must return a torch.Tensor
        # IMP: This is just for us humans, python doesn't really do anything about it

    def encode_text(self,text:str) -> torch.Tensor:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], dtype=torch.long, device=self.device)  # [1, L]

    def forward(self, input_ids: torch.Tensor):
        with torch.no_grad():
            out = self.model(input_ids= input_ids, output_hidden_states= True)
        hidden = out.hidden_states[-1]   # [B, L, D]
        logits = out.logits              # [B, L, V]
        return hidden, logits

        