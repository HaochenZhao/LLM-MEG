#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:13:48 2022

@author: blyu
"""

from argparse import ArgumentParser
import os,os.path
import numpy as np
import mne
from tqdm import tqdm
from mne.io import read_info, write_fiducials, read_fiducials
from mne.coreg import Coregistration
import matplotlib.pyplot as plt
from scipy.io import loadmat
from mne.minimum_norm import apply_inverse, make_inverse_operator, apply_inverse_epochs, write_inverse_operator, read_inverse_operator
from mne.beamformer import apply_lcmv, make_lcmv, apply_lcmv_epochs
from mne.datasets import sample
import time
import multiprocessing as mp 
# %matplotlib inline

from argparse import ArgumentParser

argp = ArgumentParser()
argp.add_argument('--ses', default='1')
args = argp.parse_args()

def run():

    preproc_dir = "/data/zhiang/holmes/preproc_new_upload"
    subjects_dir = "/data/zhiang/holmes_test/smri"
    subjs = [1,2,3] 
    sesssions = [1,2,3,4,5,6,7,8,9,10]
    epochs = ['wdonset'] # ,'wdoffset'
    epochs_plot_twin = dict()
    epochs_plot_twin['wdonset'] = [-0.1,0.3]
    epochs_plot_twin['wdoffset'] = [-0.3,0.1]

    plot_src = 0


    #for epoch in epochs:
    epoch = 'wdonset'
        #for sub in subjs:  
    sub = 1
            #for ses in sesssions:   
    ses = int(args.ses)

    subj_path = f"{preproc_dir}/sub-{sub:03d}_ses-{ses:03d}_clean_raw/"
    subj_src_path = f"{preproc_dir}/sub-{sub:03d}_ses-{ses:03d}_clean_raw/s/"
    subj_data_file = os.path.join(subj_path,f"sub-{sub:03d}_ses-{ses:03d}_clean_preproc_raw.fif")
    subj_epoch_file = os.path.join(subj_path,f"sub-{sub:03d}_ses-{ses:03d}_epoch_{epoch}-epo.fif")
    subj_bl_file = os.path.join(subj_path,f"sub-{sub:03d}_ses-{ses:03d}_clean_preproc_bl_raw.fif")
    subj_tag = f"sub-{sub:03d}_ses-{ses:03d}"

    if not os.path.exists(subj_src_path):
        os.mkdir(subj_src_path)

    if True:

        data_bl = mne.io.read_raw_fif(subj_bl_file)

        epochs = mne.read_epochs(subj_epoch_file)

        # evoked.data [sensor x time] can be replaced by other date with the same shape  
        evoked = epochs.average().pick("mag")

        inv_fn = os.path.join(subj_src_path,f"inv_ico4_dSPM_{epoch}.fif")

        if not os.path.exists(inv_fn):
            noise_cov_fn = os.path.join(subj_src_path,'bl-cov.fif')
            if not os.path.exists(noise_cov_fn):
                noise_cov = mne.compute_raw_covariance(data_bl,reject_by_annotation=True, picks='mag')
                noise_cov.save(noise_cov_fn)
            else:
                noise_cov = mne.read_cov(noise_cov_fn)

            src = mne.setup_source_space(f"holmes_s{sub:03d}", spacing="oct6", add_dist="patch", subjects_dir=subjects_dir)
            fwd = mne.read_forward_solution(os.path.join(subj_path+f"sub-{sub:03d}_ses-{ses:03d}_epoch_{epoch}-ico4-fwd.fif"))

            ## source localization using dSPM
            # evoked data
            inverse_operator = make_inverse_operator(evoked.info, fwd, noise_cov, loose=0.2, depth=0.8)

            write_inverse_operator(inv_fn,inverse_operator)

        else:
            inverse_operator = read_inverse_operator(inv_fn)


        
        method = "dSPM"
        snr = 3.0
        lambda2 = 1.0 / snr**2
        '''
        stc, residual = apply_inverse(
            evoked, #
            inverse_operator,
            lambda2,
            method=method,
            pick_ori=None,
            return_residual=True,
            verbose=True,
        )   

        stc.save(os.path.join(subj_src_path,epoch+'_evoked_dSPM'),overwrite=True)

        if plot_src:
            for hs in ['lh','rh']: 
                vertno_max, time_max = stc.get_peak(hemi=hs)

                # if hs == 'lh':
                #     lims = [stc.data[0:4095, :].mean(), stc.data[0:4095, :].max()/2, stc.data[0:4095, :].max()]
                # elif hs == 'rh':
                #     lims = [stc.data[4096:, :].mean(), stc.data[4096:, :].max()/2, stc.data[4096:, :].max()]

                surfer_kwargs = dict(
                    hemi=hs,
                    subjects_dir=subjects_dir,
                    # clim=dict(kind="value", lims=lims),
                    clim=dict(kind="percent", pos_lims=[0,97.5,100]),
                    views="lateral",
                    initial_time=time_max,
                    time_unit="s",
                    size=(800, 800),
                    smoothing_steps=10,
                )
                mne.viz.set_3d_backend('pyvistaqt')
                brain = stc.plot(**surfer_kwargs)
                brain.add_foci(
                    vertno_max,
                    coords_as_verts=True,
                    hemi=hs,
                    color="blue",
                    scale_factor=0.6,
                    alpha=0.5,
                )
                brain.add_text(
                    0.1, 0.9, "dSPM (plus location of maximal activation)", "title", font_size=14
                )
                brain.save_image(os.path.join(subj_src_path,epoch+'_evoked_dSPM-'+hs+'.png'))
                brain.close()        
        
        # epoched data (may be extremely memory consuming, consider crop epoch length, or do it one sample point a time)    
        # epochs_ = epochs[0:10] # take the first 10 epochs as an example
        
        stcs = apply_inverse_epochs(
            epochs_,
            inverse_operator,
            lambda2,
            method,
            label=None,
            pick_ori="normal",
            nave=evoked.nave,
        )
        #stcs.save(os.path.join(subj_src_path,epoch+'_epoch_dSPM'),overwrite=True)
        '''

        
        # provide a 'label' to extract data from a specified roi
        # see below an example roi (i.e., left primary auditory cortex) imported from MNE sample data, which is not consistent with the current source space
        # https://mne.tools/stable/auto_examples/inverse/compute_mne_inverse_epochs_in_label.html#sphx-glr-auto-examples-inverse-compute-mne-inverse-epochs-in-label-py

        # data_path = sample.data_path()
        # meg_path = data_path / "MEG" / "sample"

        data = np.load('/data/zhiang/hidden_states/iter_0420000/sub-{:03d}/pca_200_hs_{}.npy'.format(sub, ses))
        (aa,bb) = data.shape
        print(data.shape)
        a = []
        if aa%10 == 0:
            l = int(aa/10)
        else:
            l = int(aa/10+1)
        with tqdm(total=l) as pbar:
            for i in range(l):
                epochs_ = epochs[i*10:i*10+10]
                stcs = apply_inverse_epochs(
                    epochs_,
                    inverse_operator,
                    lambda2,
                    method,
                    label=None,
                    pick_ori="normal",
                    nave=evoked.nave,
                )
                for stc in stcs:
                    a.append(stc.data)
                pbar.update()

        a = np.array(a)
        print(a.shape)
        np.save('/data/zhiang/data/extracted/all/{:03d}-{:03d}_wdonset_ico4_dSPM.npy'.format(sub, ses), a)


        '''
        #stcs.save(os.path.join(subj_src_path,epoch+'_epcoh_roi_dSPM'),overwrite=True) # source time courses of all vertices in this roi
        '''

        '''
        # compute sign flip to avoid signal cancellation when averaging signed values
        mean_stc = sum(stcs) / len(stcs)
        flip = mne.label_sign_flip(label, inverse_operator["src"])
        label_mean_flip = np.mean(flip[:, np.newaxis] * mean_stc.data, axis=0) # source time course of this roi (averaged across vertices)
        '''


        '''
        ## source localization using LCMV
        cov_t1 = 0.01
        cov_t2 = 0.4

        data_cov_fn = os.path.join(subj_src_path,epoch+'_tw_'+str(cov_t1)+'_'+str(cov_t2)+'_epoch-cov.fif')
        if not os.path.exists(data_cov_fn):
            data_cov = mne.compute_covariance(epochs, tmin=cov_t1, tmax=cov_t2, method="empirical")            
            data_cov.save(data_cov_fn)
        else:
            data_cov = mne.read_cov(data_cov_fn)

        # noise_cov_epoch = mne.compute_covariance(epochs, tmin=-0.1, tmax=-0.01, method="empirical")            
        noise_cov = mne.compute_raw_covariance(data_bl,reject_by_annotation=True, picks='mag')

        data_rank = mne.compute_rank(data_cov,info=epochs.info)
        noise_rank = mne.compute_rank(noise_cov,info=data_bl.info)

        # evoked data
        filters = make_lcmv(
            evoked.info,
            fwd,
            data_cov,
            noise_cov=noise_cov,
            pick_ori="max-power",
            weight_norm="unit-noise-gain",
            rank={'mag':np.min([data_rank['mag'],noise_rank['mag']])},
        )

        stc = apply_lcmv(evoked, filters)
        stc.save(os.path.join(subj_src_path,epoch+'_evoked_lcmv'),overwrite=True)
        '''

        '''
        if plot_src:
            for hs in ['lh','rh']:         
                vertno_max, time_max = stc.get_peak(hemi=hs)

                surfer_kwargs = dict(
                    hemi=hs,
                    subjects_dir=subjects_dir,
                    clim=dict(kind="percent", pos_lims=[0,97.5,100]),
                    views="lateral",
                    initial_time=time_max,
                    time_unit="s",
                    size=(800, 800),
                    smoothing_steps=10,
                )
                brain = stc.plot(**surfer_kwargs)
                brain.add_foci(
                    vertno_max,
                    coords_as_verts=True,
                    hemi=hs,
                    color="blue",
                    scale_factor=0.6,
                    alpha=0.5,
                )

                brain.add_text(
                    0.1, 0.9, "LCMV (plus location of maximal activation)", "title", font_size=14
                )
                brain.save_image(os.path.join(subj_src_path,epoch+'_evoked_lcmv-'+hs+'.png'))
                brain.close() 

        # epoched data (may be extremely memory consuming, consider crop epoch length, or do it one sample point a time)    
        epochs_ = epochs[0:10] # take the first 10 epochs as an example
        stcs = apply_lcmv_epochs(epochs_.pick(picks='mag'), filters)
        stc.save(os.path.join(subj_src_path,epoch+'_epochs_lcmv'),overwrite=True)
        '''

        '''
        a = np.zeros((0, 28, 301))

        for i in range(860):
            print(i)
            epochs_ = epochs[i*10:i*10+10]
            stcs = apply_lcmv_epochs(epochs_.pick(picks='mag'), filters)

            for stc in stcs:
                s = stc.data
                s = s.reshape((1, 28, 301))
                a = np.concatenate((a, s), axis=0)

        np.save(os.path.join(subj_src_path,epoch+'_epcoh_lcmv'), a)
        '''


def maket(t):
    t = time.localtime(t)
    t = time.strftime("%Y-%m-%d %H:%M:%S", t)
    return t

if __name__ == '__main__':
    start_t = int(time.time())
    print("session{} start at ".format(int(args.ses)), maket(start_t))

    run()

    end_t = int(time.time())
    print("end at ", maket(end_t))

    used_t = maket(end_t-start_t)
    print("time used: ", used_t)
