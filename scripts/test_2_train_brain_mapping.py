import os, mne
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from argparse import ArgumentParser
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict
import wandb

from model.brain_mapping_linear import LinearMapping

from string import punctuation

argp = ArgumentParser()
argp.add_argument("--learning_rate", default='1e-5')
argp.add_argument("--sample_index", default='175')
argp.add_argument("--hidden_layer", default='15')
argp.add_argument("--seed", default='1')
argp.add_argument("--gpu", default='0')
args = argp.parse_args()


# Set parameters
learning_rate = float(args.learning_rate)
sample_index = int(args.sample_index)
hidden_layer = int(args.hidden_layer)
seed = int(args.seed)
gpu_id = args.gpu
l2_lambda = 1e-5

# Move model and tokenizer to CUDA if available
device = torch.device("cuda:"+gpu_id if torch.cuda.is_available() else "cpu")

# Set wandb logging
wandb.init(project="my_project")
wandb.config.device = 'gpu_'+gpu_id
wandb.config.seed= seed
wandb.config.learning_rate = learning_rate
wandb.config.sample_index = sample_index
wandb.config.hidden_layer = hidden_layer
wandb.config.l2_lambda = l2_lambda

# Set seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

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

def main():

    mapping = LinearMapping(4096, 269).to(device)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(mapping.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    epoch = 96
    steps = 200
    batch_size = 10

    wandb.watch(mapping, log='all')

    train(mapping, epoch, steps, batch_size, loss_fn, optimizer)


def train(model, epoch, steps, batch_size, loss_fn, optimizer):
    global line_counts
    global word_counts
    global l_end
    global r_end

    for i in range(epoch):
        print("--------epoch {}--------".format(i+1))

        model.train()
        line_counts = 0
        word_counts = 0
        l_end = 0
        r_end = 0
        for step in range(steps):
            [activations, targets] = get_train_data(batch_size)
            outputs = model.forward(activations)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if step % (steps/10) == 0:
                print("train step: {}, loss: {}".format(step, loss.item()))
                wandb.log({
                    "train_loss": loss.item()
                },
                step+i*steps)
        
        model.eval()
        total_test_loss = 0
        corr_list = []
        with torch.no_grad():
            [activations, targets] = get_test_data()
            for ii, (activation, target) in enumerate(zip(activations, targets)):
                output = model.forward(activation)
                loss = loss_fn(output, target)
                total_test_loss += loss.item()
                #accuracy = (output == target).sum()*100/len(target)
                #total_accuracy += accuracy
                o = output.cpu()
                t = target.cpu()
                corr = stats.pearsonr(o, t)
                corr_list.append(corr.statistic)

        print("epoch {}, test loss: {}, test correlation: {}".format(i+1, total_test_loss, np.mean(corr_list)))
        wandb.log({
            "test_loss": total_test_loss,
            "test_correlation": np.mean(corr_list)
        })

        #torch.save(model, "{}/mapping_{}.pth".format(experiment_dir, i+1))
        wandb.save("mapping_{}.h5".format(i+1))
        print("save epoch{}".format(i+1))
    

def get_train_data(num):
    global line_counts
    global word_counts
    global l_end
    global r_end

    # get lines from txt file
    context = []
    for index, line in enumerate(open('/sshare/home/yanzhiang/data/train.txt')):
        line_list = list(line)
        line_list[len(line_list)-1] = ' '
        line_new = ''.join(line_list)
        context.append(line_new)
    context_total = len(context)

    # joint txt with lines for model
    txt = ''
    end_flag = False    
    for i in range(num):
        index = line_counts+i
        txt += context[index]
        if index == context_total-1:
            end_flag = True
            break
    if end_flag:
        line_counts = 0
    else:
        line_counts += num
    
    #print(line_counts)

    txt = txt.replace('-', ' ')

    # s for hidden activations
    s = get_layer_states_by_text(hidden_layer, txt)

    #print(s.shape)

    for c in punctuation_string:
        txt = txt.replace(c, '')
    words = txt.split()

    l_end = r_end
    r_end = r_end + len(words)

    # check for dropped words
    drop_index = []
    for i in range(len(words)):
        if word_counts+i in wd_id_dropped:
            drop_index.append(i)
    l_drop = len(drop_index)
    if l_drop != 0:
        #print("catch dropped")
        r_end -= l_drop
        (x,y) = s.shape
        new_s = torch.zeros(x-len(drop_index), y).to(device)
        t = 0
        for i in range(x):
            if i not in drop_index:
                new_s[t] = s[i]
                t += 1
        s = new_s
    
    #print(l_end,":",r_end)


    word_counts += len(words)


    meg_signals = get_MEG_signals()
    e = meg_signals[l_end:r_end,:,sample_index:sample_index+1]
    (x,y,z) = e.shape
    e = e.reshape((x,y))

    m = torch.from_numpy(e).to(device)
    #print(m.shape)
    #exit(0)

    if end_flag:
        #print("total words:", word_counts)
        word_counts = 0
        l_end = 0
        r_end = 0

    return [s.float(), m.float()]

def get_test_data():
    txt = ''
    for index, line in enumerate(open('/sshare/home/yanzhiang/data/test.txt')):
        line_list = list(line)
        line_list[len(line_list)-1] = ' '
        line_new = ''.join(line_list)
        txt += line_new

    txt = txt.replace('-', ' ')

    s = get_layer_states_by_text(hidden_layer, txt)

    meg_signals = get_MEG_signals()
    e = meg_signals[-1717:,:,sample_index:sample_index+1]
    (x,y,z) = e.shape
    e = e.reshape((x,y))

    m = torch.from_numpy(e).to(device)

    return [s.float(), m.float()]


def get_layer_states_by_text(layer, input_text):
    line = input_text
    line = line.strip()
    inputs = tokenizer(line, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
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
    words_num = words_len - len(non_words_index)
    untokenized_sent = tuple(words)
    mapping = match_tokenized_to_untokenized(tokens, untokenized_sent)

    with torch.no_grad(): # No need to compute gradients
        # Forward pass through the model with output_hidden_states=True
        outputs = target_model(**inputs, output_hidden_states=True)
        # Extract hidden states
        hidden_states = outputs.hidden_states
        #print(hidden_states)
        target_hidden_layers = hidden_states[layer][0]

        layer_activations = torch.zeros(words_num, target_hidden_layers.shape[1]).to(device)
        words_count = 0
        for index in range(words_len):
            if index in non_words_index:
                continue
            for i in mapping[index]:
                layer_activations[words_count] += target_hidden_layers[i]
            layer_activations[words_count] /= len(mapping[index])
            words_count += 1

    return layer_activations

def get_MEG_signals():

    return meg_data

if __name__ == '__main__':
    global target_model
    global tokenizer
    global punctuation_str

    global wd_id_dropped
    global meg_data

    model_name = 'iter_0420000'
    model_dir = '/sshare/pku-ai4s/valm-6b-checkpoints/'

    local_model_directory = model_dir+model_name
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(local_model_directory)
    target_model = AutoModel.from_pretrained(local_model_directory)
    target_model.eval().to(device)

    punctuation_string = punctuation

    data_path = '/sshare/home/yanzhiang/data/holmes_test'
    epoch_fn = 'sub-001_ses-001_epoch_wdonset-epo.fif' # -200:5:1000 ms, 250Hz sampling rate, aligned to word onset

    epochs = mne.read_epochs(os.path.join(data_path,epoch_fn))

    wd_id_all = epochs.event_id
    wd_id_dropped = [index for index, item in enumerate(epochs.drop_log) if item]

    meg_data = epochs.get_data(picks=["mag"]) * 2e12
    l_end = 0
    r_end = 0
    
    main()