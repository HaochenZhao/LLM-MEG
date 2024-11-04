#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:08:39 2020

@author: blyu
"""

import torch, h5py, os
import numpy as np
from argparse import ArgumentParser
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict

def match_tokenized_to_untokenized(tokenized_sent, untokenized_sent):

    mapping = defaultdict(list)
    untokenized_sent_index = 0

    for tokenized_sent_index in range(len(tokenized_sent)):
        if tokenized_sent_index == 0:
          mapping[untokenized_sent_index].append(tokenized_sent_index)
          i1 = tokenized_sent_index
        else:
          tmp = ''.join(tokenized_sent[i1:tokenized_sent_index+1])
          if tmp in '▁'+untokenized_sent[untokenized_sent_index]:
              mapping[untokenized_sent_index].append(tokenized_sent_index)
          else:
              untokenized_sent_index += 1
              mapping[untokenized_sent_index].append(tokenized_sent_index)
              i1 = tokenized_sent_index
              
    return mapping
              
argp = ArgumentParser()
argp.add_argument('input_path')
argp.add_argument('model_name')
argp.add_argument('layer')
argp.add_argument('gpuid')
args = argp.parse_args()

layer = int(args.layer)

model_dir = '/sshare/models/valm-6b-checkpoints/'

h5fn = args.input_path+'.'+args.model_name+'.l'+str(layer)+'.hdf5'
if os.path.exists(h5fn):
    # Delete the file
    os.remove(h5fn)
    print(f"The file {h5fn} has been deleted.")
    print(f"Converting: {h5fn}")
else:
    print(f"Converting: {h5fn}")
    
    
local_model_directory = model_dir+args.model_name

if args.model_name == '01-valm1b3':
    LAYER_COUNT = 23
    FEATURE_COUNT = 2048
elif args.model_name[0:4] == 'iter':
    LAYER_COUNT = 32
    FEATURE_COUNT = 4096
else:
    raise ValueError("Invalid model name")
  
# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_dir+args.model_name)
model = AutoModel.from_pretrained(model_dir+args.model_name)
model.eval()

# Move model and tokenizer to CUDA if available
device = torch.device("cuda:"+args.gpuid if torch.cuda.is_available() else "cpu")
model.to(device)

for index, line in enumerate(open(args.input_path+'.txt')):
    
    # Tokenize the input
    line = line.strip()
    inputs = tokenizer(line, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])    
    
    words = line.split()
    untokenized_sent = tuple(words)
    mapping = match_tokenized_to_untokenized(tokens, untokenized_sent)


    with torch.no_grad(): # No need to compute gradients
        # Forward pass through the model with output_hidden_states=True
        outputs = model(**inputs, output_hidden_states=True)

        # Extract hidden states
        hidden_states = outputs.hidden_states
        
        with h5py.File(h5fn, 'a') as fout:
           dset = fout.create_dataset(str(index), (1, len(untokenized_sent), FEATURE_COUNT))
           tmp = hidden_states[layer][0,:,:].cpu().numpy()
           tmp = torch.tensor([np.mean(tmp[mapping[i][0]:mapping[i][-1]+1,:], axis=0) for i in range(len(untokenized_sent))])
           tmp = tmp.unsqueeze(0)
           dset[:,:,:] = np.array(tmp)

