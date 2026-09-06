import torch

#Why this function exists and naiver comparing fails:
    # When we get sentence piece as tokens, we predict the next word from data we have seen already
    # For the first token, there is no previous data to predcit from, harms NLL
    # For the last token, there is no target to compare against, harms NLL
    # Hence this function exists for dropping those cases

def shift_for_next_token(hidden,logits,input_ids):
    h_pred = hidden[:, :-1, :] # chop the last position off
    lgts_pred = logits[:,:-1, :] #chop the last from logits as well
    y_target = input_ids [:, 1:] #Chop the first position and shift
    return h_pred, lgts_pred, y_target


def true_token_stats(lgts_pred, y_target):
    probs  = torch.softmax(lgts_pred.float(), dim=-1)          # [B, L-1, V] — turn scores into probabilities
    p_true = probs.gather(-1, y_target.unsqueeze(-1)).squeeze(-1)  # [B, L-1] — pull out just the true token's probability
    nll    = -torch.log(p_true + 1e-12)                        # [B, L-1] — how surprised GPT was
    return p_true, nll

def collect_chunks(dataset, tokenizer, seq_len, n_chunks_needed):
    buffer = []
    chunks = []

    for story in dataset:
        ids = tokenizer.encode(story["text"], add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)
        buffer.extend(ids)

        # buffer[start:end], starting from starting index upto end but not end 
        
        while len(buffer) >= seq_len:
            chunks.append(buffer[:seq_len]) #Add seq len upto buffer
            buffer = buffer[seq_len:] # Adding the colon at the end acts as removing from the buffer

        if len(chunks) >= n_chunks_needed:
            break

    return chunks

def predictive_entropy(lgts_pred):
    """Entropy of GPT's own full next-token distribution — doesn't require
    knowing the true token, unlike NLL. High entropy = GPT is unsure; low = confident."""
    probs = torch.softmax(lgts_pred.float(), dim=-1)  # [B, L-1, V]
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)  # [B, L-1]
    return entropy

