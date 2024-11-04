#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:08:39 2020

@author: blyu
"""

# python txt2hidden_states_yi.py /sshare/home/lyubingjiang/lm/corpus/ptb/ptb3-wsj-23  iter_0010000  0      1
#                                input_txt_file                                       model         layer  gpu_id

import torch, os, string
import numpy as np
from argparse import ArgumentParser
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
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
          print(tmp, '\x20\x20▁'+untokenized_sent[untokenized_sent_index])
          if tmp in '▁'+untokenized_sent[untokenized_sent_index]:
              print(1)
              mapping[untokenized_sent_index].append(tokenized_sent_index)
          else:
              print(0)
              untokenized_sent_index += 1
              mapping[untokenized_sent_index].append(tokenized_sent_index)
              i1 = tokenized_sent_index
          #print(mapping)
              
    return mapping

argp = ArgumentParser()
argp.add_argument('--input_path', default=None)
argp.add_argument('--model_name', default='iter_0420000')
argp.add_argument('--layer', default='8')
argp.add_argument('--gpuid', default='0')
argp.add_argument('--display', default=True)
args = argp.parse_args()

is_display = args.display

punctuation_string = string.punctuation

if args.model_name == 'bert-base-cased':
    LAYER_COUNT = 12
    FEATURE_COUNT = 768
elif args.model_name == 'bert-large-cased':
    LAYER_COUNT = 24
    FEATURE_COUNT = 1024
elif args.model_name == '01-valm1b3':
    LAYER_COUNT = 23
    FEATURE_COUNT = 2048
elif args.model_name[0:4] == 'iter':
    LAYER_COUNT = 32
    FEATURE_COUNT = 4096
else:
    raise ValueError("Invalid model name")

def main():

    layer = int(args.layer)
    model_name = args.model_name
    gpuid = args.gpuid
    input_path = args.input_path

    if is_display:
        print('start running...')

    if input_path is not None:
        return get_layer_states_by_path(layer, model_name, gpuid, input_path)
    else:
        if is_display:
            print('please input:')
        input_text = input()
        return get_layer_states_by_text(layer, model_name, gpuid, input_text)

def get_layer_states_by_path(layer, model_name, gpuid, input_path):
    model_dir = '/data/minghua/hf_output/20240116-1900/'
    '''
    h5fn = args.input_path+'.'+args.model_name+'.l'+str(layer)+'.hdf5'
    if os.path.exists(h5fn):
        # Delete the file
        os.remove(h5fn)
        print(f"The file {h5fn} has been deleted.")
        print(f"Converting: {h5fn}")
    else:
        print(f"Converting: {h5fn}")
    '''
    local_model_directory = model_dir+model_name
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(local_model_directory)
    model = AutoModel.from_pretrained(local_model_directory)
    model.eval()
    # Move model and tokenizer to CUDA if available
    device = torch.device("cuda:"+gpuid if torch.cuda.is_available() else "cpu")
    model.to(device)

    for index, line in enumerate(open(input_path+'.txt')):
        # Tokenize the input
        line = line.strip()
        inputs = tokenizer(line, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])    

        words = line.split()
        # todo: delete punctuations
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



def get_layer_states_by_text(layer, model_name, gpuid, input_text):
    model_dir = '/data/minghua/hf_output/20240116-1900/'
    local_model_directory = model_dir+model_name
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(local_model_directory)
    model = AutoModel.from_pretrained(local_model_directory)

    # Load model directly
    #tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    #model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    model.eval()
    # Move model and tokenizer to CUDA if available
    device = torch.device("cuda:"+gpuid if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    line = input_text
    line = line.strip()
    inputs = tokenizer(line, return_tensors="pt").to(device)
    if is_display:
        print('your input:')
        print(inputs)
    input_ids = inputs["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    new_tokens = []
    for token in tokens:
        new_token = token.replace('Ġ', '▁')
        new_tokens.append(new_token)
    tokens = new_tokens
    if is_display:
        print("tokens:")
        print(tokens)
    non_words_index = []
    words = line.split()
    words_len = len(words)
    for i in range(words_len):
        flag = True
        for j in range(len(words[i])):
            if words[i][j] not in punctuation_string:
                flag = False
        if flag:
            non_words_index.append(i)
    if is_display:
        print("non-word index:")
        print(non_words_index)
    words_num = words_len - len(non_words_index)
    untokenized_sent = tuple(words)
    if is_display:
        print("words:")
        print(untokenized_sent)
    mapping = match_tokenized_to_untokenized(tokens, untokenized_sent)
    if is_display:
        print("mapping:")
        print(mapping)
    with torch.no_grad(): # No need to compute gradients
        # Forward pass through the model with output_hidden_states=True
        outputs = model(**inputs, output_hidden_states=True)
        # Extract hidden states
        hidden_states = outputs.hidden_states
        if is_display:
            print('hidden states:')
            print(len(hidden_states))
        #print(hidden_states)
        target_hidden_layers = hidden_states[layer][0]
        if is_display:
            print(target_hidden_layers.shape)

        layer_activations = torch.zeros(words_num, target_hidden_layers.shape[1]).to(device)
        if is_display:    
            print(layer_activations.shape)
        words_count = 0
        for index in range(words_len):
            if index in non_words_index:
                continue
            if is_display:
                print(mapping[index])
            for i in mapping[index]:
                layer_activations[words_count] += target_hidden_layers[i]
            layer_activations[words_count] /= len(mapping[index])
            words_count += 1
        if is_display:
            print(words_count)
            print('result:')
            print(layer_activations)
        return layer_activations
    

if __name__ == '__main__':
    main()

        



