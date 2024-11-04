import numpy as np
import mne, os


def align():
    for i in range(1,11,1):
        data = np.load('/data/zhiang/hidden_states/iter_0420000/total/hs_{}.npy'.format(i))

        data_path = '/data/zhiang/holmes/preproc_new_upload/sub-001_ses-{:03d}_clean_raw'.format(i)
        epoch_fn = 'sub-001_ses-{:03d}_epoch_wdonset-epo.fif'.format(i)
        epochs = mne.read_epochs(os.path.join(data_path,epoch_fn))
        wd_id_all = epochs.event_id
        wd_id_dropped = [index for index, item in enumerate(epochs.drop_log) if item]
        meg_data = epochs.get_data(picks=["mag"]) * 2e12
        meg_data = np.array(meg_data)
        (a,b,c) = meg_data.shape
        (d,e) = data.shape
        print(d, a+len(wd_id_dropped))

def check():
    s = 3
    i = 8
    data_path = '/data/zhiang/holmes/preproc_new_upload/sub-{:03d}_ses-{:03d}_clean_raw'.format(s,i)
    epoch_fn = 'sub-{:03d}_ses-{:03d}_epoch_wdonset-epo.fif'.format(s,i)
    epochs = mne.read_epochs(os.path.join(data_path,epoch_fn))
    print(epochs.info)
    print(epochs.drop_log)
    meg_data = epochs.get_data(picks=["mag"])
    meg_data = np.array(meg_data)
    print(meg_data.shape)

def ltest():
    data = np.load('/data/zhiang/hidden_states/iter_0420000/sub-002/pca_200_hs_1.npy')
    data = np.load('/data/zhiang/data/extracted/all/002-002_wdonset_ico4_dSPM.npy')

    print(data.shape)


def test_alpha():
    data = np.load('/data/zhiang/data/search_alpha/iter_0100000/sub-002/ba_2.npy')
    print(data)
    print(data.shape)

if __name__ == '__main__':
    data = np.load('/data/zhiang/hidden_states/iter_0420000/layer8/total/e_hs_2.npy')
    print(data.shape)