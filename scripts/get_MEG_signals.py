#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import mne
import os
import numpy as np
from argparse import ArgumentParser

argp = ArgumentParser()
argp.add_argument('--display', default=True)
args = argp.parse_args()

is_display = args.display

def main():
    print('start running...')
    get_MEG_signals()

def get_MEG_signals():

    data_path = '/data/zhiang/holmes/preproc_new_upload/sub-003_ses-002_clean_raw'
    epoch_fn = 'sub-003_ses-002_epoch_wdonset-epo.fif' # -200:5:1000 ms, 250Hz sampling rate, aligned to word onset

    epochs = mne.read_epochs(os.path.join(data_path,epoch_fn))
    #print(epochs)
    #epochs.plot_drop_log()
    #exit(0)

    if is_display:
        print('reading finished')

    # downsample to reduce data size if necessary
    # epochs_re = epochs.resample(sfreq=100) # to 100Hz
    # epoch_fn_new = 're_sub-001_ses-001_epoch_wdonset-epo.fif'
    # epochs.save(os.path.join(data_path,epoch_fn_new))
    
    wd_id_all = epochs.event_id
    wd_id_dropped = [index for index, item in enumerate(epochs.drop_log) if item]
    print(wd_id_dropped)
    print(len(wd_id_dropped))
    

    meg_data = epochs.get_data(picks=["mag"]) * 2e12 # word x channel x time, does not include dropped words
    meg_data = np.array(meg_data)
    print(meg_data.shape)
    exit(0)

    #print(meg_data)
    if is_display:
        print('shape:')
        print(len(meg_data))
        #print(wd_id_dropped)
        print(len(wd_id_dropped))
        #exit(0)
        '''
        ma = 0
        mi = 0
        for _ in meg_data:
            for i in _:
                for j in i:
                    if j < mi:
                        mi = j
                    if j > ma:
                        ma = j
        
        '''
        ma = 1.6042434532856955e-12
        mi = -1.8107518064429671e-12
        print(ma, mi)

    #meg_signals = meg_data[:,:,175:176]
    #(x,y,z) = meg_signals.shape
    #meg_signals = meg_signals.reshape((x,y))

    meg_signals = meg_data

    print(meg_signals.shape)

    #np.save("meg_data.npy", meg_signals)
    #np.savetxt("meg_signals.txt", meg_signals)

if __name__ == '__main__':
    main()