from string import punctuation

punctuation_str = punctuation

line_num = [0, 650, 547, 383, 618, 446, 561, 516, 592, 479, 508]

_2_num = [0,60,73,79,96,61,87,91]

index = 2

nu = []

for e in range(1,8,1):
    num = _2_num[e]

    line_counts = 0
    word_counts = 0
    context = []
    #path = '/data/zhiang/txt/total/{:>02d}_total.txt'.format(index)
    path = '/data/zhiang/txt/{:>02d}_{}_t.txt'.format(index, e)
    for i, line in enumerate(open(path)):
        context.append(line)
    context_total = len(context)
    total_words = []

    for i in range(context_total):
        txt = context[i]

        txt = txt.replace('-', ' ')
        #txt = txt.replace('\'', ' \' ')
        for c in punctuation_str:
            txt = txt.replace(c, '')
        words = txt.split()
        word_counts += len(words)
        total_words = total_words + words
    #print(total_words)
    nu.append(len(total_words))
    print(total_words)
    print(len(total_words), sum(nu))

#print(sum(nu))

line_counts = 0
word_counts = 0
context = []
path = '/data/zhiang/txt/total/{:>02d}_total.txt'.format(index)
for i, line in enumerate(open(path)):
    context.append(line)
context_total = len(context)
total_words = []
txt = ''
for i in range(context_total):
    txt = txt + context[i]
txt = txt.replace('-', ' ')
    #txt = txt.replace('\'', ' \' ')
for c in punctuation_str:
    txt = txt.replace(c, '')
words = txt.split()
word_counts += len(words)
total_words = total_words + words
#print(total_words)
print(len(total_words))

print('finish')
exit(0)