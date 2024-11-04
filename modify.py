import numpy as np
from string import punctuation

punctuation_str = punctuation

line_num = [0, 650, 547, 383, 618, 446, 561, 516, 592, 479, 508]

modi_lists = [[],[],[1561],[105,6979, 6982],[3206],[],[],[],[],[],[5326]]

def run(modi_ses):
    name = "iter_0010000"
    modi_list = modi_lists[modi_ses]

    txt = "/data/zhiang/txt/total/{:>02d}_total.txt".format(modi_ses)
    num  = line_num[modi_ses]

    line_counts = 0
    word_counts = 0
    context = []
    for i, line in enumerate(open('/data/zhiang/txt/total/{:>02d}_total.txt'.format(modi_ses))):
        context.append(line)
    context_total = len(context)
    total_words = []

    for i in range(num):
        txt = context[i]

        txt = txt.replace('-', ' ')
        #txt = txt.replace('\'', ' \' ')
        for c in punctuation_str:
            txt = txt.replace(c, '')
        words = txt.split()
        word_counts += len(words)
        total_words = total_words + words

    for i in modi_list:
        print(total_words[i])
    data = np.load("/data/zhiang/hidden_states/{}/total/e_hs_{}.npy".format(name,modi_ses))
    print(data.shape)
    data = np.delete(data, modi_list, axis=0)
    print(data.shape)
    np.save("/data/zhiang/hidden_states/{}/total/hs_{}.npy".format(name,modi_ses), data)


if __name__ == '__main__':
    for i in [2,3,4,10]:
        run(i)