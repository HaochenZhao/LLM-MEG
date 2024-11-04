import os, mne
import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from transformers import AutoModel, AutoTokenizer

from model.brain_mapping_linear import LinearMapping

def main():
    model_dir = '/sshare/models/valm-6b-checkpoints/'
    model_path = model_dir + 'iter_0420000'
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()
    # Move model and tokenizer to CUDA if available
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data_path = '/sshare/home/yanzhiang/data/homles'
    epoch_fn = 'sub-001_ses-001_epoch_wdonset-epo.fif' # -200:5:1000 ms, 250Hz sampling rate, aligned to word onset

    epochs = mne.read_epochs(os.path.join(data_path,epoch_fn))

    wd_id_all = epochs.event_id
    wd_id_dropped = [index for index, item in enumerate(epochs.drop_log) if item]

    meg_data = epochs.get_data(picks=["mag"]) # word x channel x time, does not include dropped words

    mapping = LinearMapping(4096, 269).to(device)
    loss_list = []
    c_list = []
    count = 0
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(mapping.parameters(), lr=0.005, weight_decay=0.01)

    mapping.train()

    txt_path = '/sshare/home/yanzhiang/data/homles/01_1_t.txt'
    word_num = 0
    for index, line in enumerate(open(txt_path)):
        # Tokenize the input
        line = line.strip()
        inputs = tokenizer(line, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        length = len(input_ids[0])
        word_num += length
        # tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        with torch.no_grad(): # No need to compute gradients
            # Forward pass through the model with output_hidden_states=True
            outputs = model(**inputs, output_hidden_states=True)
            # Extract hidden states
            hidden_states = outputs.hidden_states
        hidden_layer = hidden_states[15]
        meg_states = meg_data[word_num].T
        #print(hidden_layer.shape)ß
        #print(meg_states.shape)
        #exit()
        x = hidden_layer[0][length-1]
        x = torch.tensor(x).float().to(device)
        y = meg_states[300]
        y = torch.tensor(y).float().to(device)

        pred = mapping.forward(x)
        loss = loss_fn(pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        count+=1
        if count % 10 == 0:
            print(f'loss:{loss_value},counts:{count}')
        loss_list.append(loss_value)
        c_list.append(count)

    plt.plot(c_list,loss_list)
    plt.savefig('./loss.jpg')
    



if __name__ == '__main__':
    main()