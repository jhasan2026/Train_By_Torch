#!/usr/bin/env python
# coding: utf-8

import re


class Tokenization:
    class Dictionary:
        def __init__(self):
            self.vocab = {}

        def makeDictionary(self,text):
            raw_text = text.replace("\n"," ")
            raw_text = raw_text.lower()
            list_of_word = re.split(r'([,.:;?_!"()\']|--|\s)',raw_text)
            list_of_word2 = [word.strip() for word in list_of_word if word.strip()]
            for word in list_of_word2:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)

            self.vocab['|endoftext|'] = len(self.vocab)

            return self.vocab

    def __init__(self,train_text):
        dic = Tokenization.Dictionary()
        self.str_to_int = dic.makeDictionary(train_text) 
        self.int_to_str = {y:x for x,y in self.str_to_int.items()}

    def encode(self,text):
        text = text.lower()
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)',text)
        preprocessed = [word.strip() for word in preprocessed if word.strip()]

        ids = []
        for word in preprocessed:
            if word in self.str_to_int:
                ids.append(self.str_to_int[word])
            else:
                ids.append(self.str_to_int['|endoftext|'])

        return ids

    def decode(self, ids):
        text = (" ".join([self.int_to_str[id] for id in ids]))
        text = re.sub(r'\s+([,.:;?!"()\'])',r'\1',text)
        return text



from torch.utils.data import Dataset, DataLoader
import torch


class CustomDataset(Dataset):
    def __init__(self,text,tokenizer,max_length,stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i+max_length]
            self.input_ids.append(torch.tensor(input_chunk))     

            target_chunk = token_ids[i+1 : i+1+max_length]
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]



def create_dataloader_cutom_tokenized(text, batch_size=4, max_length=256,stride=128,shuffle=True, drop_last=True,num_workers=0):
    tokenizer = Tokenization(text)
    dataset = CustomDataset(text,tokenizer, max_length,stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

import tiktoken

def create_dataloader(text, batch_size=4, max_length=256,stride=128,shuffle=True, drop_last=True,num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = CustomDataset(text,tokenizer, max_length,stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

