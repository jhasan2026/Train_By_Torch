#!/usr/bin/env python
# coding: utf-8

# In[1]:


from LLM_Build.GPT2_small import GPT_CONFIG_124M, GPTModel, generate_text
from LLM_Build.Evaluation import calculate_loss_of_Loader, calculate_loss_of_Batch, text_to_token_ids, token_ids_to_text, train_validation_split
import torch
import torch.nn as nn


# In[2]:


def model_evaluation(train_loader, validation_loader, model, device, evaluation_itr):
    model.eval()
    
    with torch.no_grad():
        training_loss = calculate_loss_of_Loader(
            train_loader,model,device,num_of_batch=evaluation_itr
        )
        validation_loss = calculate_loss_of_Loader(
            validation_loader,model,device,num_of_batch=evaluation_itr
        )
    
    model.train()
    return training_loss, validation_loss


# In[3]:


def generate_and_print_text(model, input_text, tokenizer, device, max_token_generate):
    model.eval()
    context_size = model.positional_embedding.weight.shape[0]
    encoded_text = text_to_token_ids(input_text, tokenizer)
    
    with torch.no_grad():
        token_ids = generate_text(
            model=model,
            inputs=encoded_text,
            max_new_tokens=max_token_generate,
            context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids,tokenizer)
    print(decoded_text.replace("\n"," "))
    model.train()


def train_model(train_dataloader, validation_dataloader, model, optimizer, device, epochs, evaluation_frequency, evaluation_itr, input_text, tokenizer,max_token_generate):
    
    training_losses, validation_losses, track_token_seen = [], [], []
    
    token_seen, global_step = 0, -1
    
    for epoch in range(epochs):
        model.train()
        for input_batch, target_batch in train_dataloader:
            optimizer.zero_grad()
            loss = calculate_loss_of_Batch(
                input_batch,target_batch,model,device
            )
            loss.backward()
            optimizer.step()
            
            global_step += 1
            token_seen += input_batch.numel()
            
            if global_step % evaluation_frequency == 0:
                training_loss, validation_loss = model_evaluation(
                    train_dataloader, validation_dataloader,model,device,evaluation_itr
                )
                
                training_losses.append(training_loss)
                validation_losses.append(validation_loss)
                track_token_seen.append(token_seen)
                print(f"{epoch+1}: (Step: {global_step:06d})")
                print(f"Training Loss: {training_loss:.3f}")
                print(f"Validation Loss: {validation_loss:.3f}")
                
        generate_and_print_text(
            model, input_text, tokenizer, device, max_token_generate
        )
                
    return training_losses, validation_losses, track_token_seen


def generate_text_randomness(model, inputs, max_new_tokens, context_size, temperature, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        input_conditionals = inputs[:, -context_size] if inputs.shape[1] > context_size else inputs
        with torch.no_grad():
            logits = model(input_conditionals)
        logits = logits[:, -1, :]

        if top_k is None:
            top_logits, _ = torch.topk(logits, top_k)
            min_value = top_logits[:, -1]
            logits = torch.where(
                logits < min_value,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        if temperature > 0.0:
            logits = logits / temperature
            probabilities = torch.softmax(logits, dim=-1)
            index_of_next = torch.multinomial(probabilities, num_samples=1)
        else:
            index_of_next = torch.argmax(logits, dim=-1, keepdim=True)

        if index_of_next == eos_id:
            break
        inputs = torch.concat((inputs, index_of_next), dim=1)
    return inputs


# %%
def generate_and_print_text_randomness(model, input_text, tokenizer, device, max_token_generate, temperature,
                                       top_k=None, eos_id=None):
    model.eval()
    context_size = model.positional_embedding.weight.shape[0]
    encoded_text = text_to_token_ids(input_text, tokenizer)

    with torch.no_grad():
        token_ids = generate_text_randomness(
            model=model,
            inputs=encoded_text,
            max_new_tokens=max_token_generate,
            context_size=context_size,
            temperature=temperature,
            top_k=top_k,
            eos_id=eos_id
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


# %%
def train_model_randomness(train_dataloader, validation_dataloader, model, optimizer, device, epochs,
                           evaluation_frequency, evaluation_itr, input_text, tokenizer, max_token_generate, temperature,
                           top_k=None, eos_id=None):
    training_losses, validation_losses, track_token_seen = [], [], []

    token_seen, global_step = 0, -1

    for epoch in range(epochs):
        model.train()
        for input_batch, target_batch in train_dataloader:
            optimizer.zero_grad()
            loss = calculate_loss_of_Batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()

            global_step += 1
            token_seen += input_batch.numel()

            if global_step % evaluation_frequency == 0:
                training_loss, validation_loss = model_evaluation(
                    train_dataloader, validation_dataloader, model, device, evaluation_itr
                )

                training_losses.append(training_loss)
                validation_losses.append(validation_loss)
                track_token_seen.append(token_seen)
                print(f"{epoch + 1}: (Step: {global_step:06d})")
                print(f"Training Loss: {training_loss:.3f}")
                print(f"Validation Loss: {validation_loss:.3f}")

        generate_and_print_text_randomness(
            model, input_text, tokenizer, device, max_token_generate, temperature, top_k, eos_id
        )

    return training_losses, validation_losses, track_token_seen
            


