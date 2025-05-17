
import torch
import torch.nn as nn

def train_validation_split(text, split_ratio):
    split_at_index = int(len(text) * split_ratio)
    return text[:split_at_index], text[split_at_index:]


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor
#%%
def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())

def calculate_loss_of_Batch(input_batch, target_batch, model, device='cpu'):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calculate_loss_of_Loader(dataloader, model, device='cpu', num_of_batch=None):
    if num_of_batch == 0:
        return float('nan')
    elif num_of_batch is None:
        num_of_batch = len(dataloader)
    else:
        num_of_batch = min(len(dataloader), num_of_batch)

    total_dataloader_loss = 0
    for index, (input_batch, target_batch) in enumerate(dataloader):
        if index < num_of_batch:
            loss = calculate_loss_of_Batch(input_batch, target_batch, model=model, device=device)
            total_dataloader_loss += loss.item()
        else:
            break

    return total_dataloader_loss / (num_of_batch + 0.000001)