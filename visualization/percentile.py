import numpy as np
import matplotlib.pyplot as plt

doc_lens = [1, 2, 3]

print('25, 50, 75 percentile of {}: {}'.format('attribute',
                                               str(np.percentile(doc_lens, [25, 50, 75]))))

min_doc_len, max_doc_len = np.min(doc_lens), np.max(doc_lens)

fig, axes = plt.subplots(1, 1)
axes.set_xlim(xmin=0, xmax=max_doc_len)
axes.hist(doc_lens, bins=max_doc_len)
plt.savefig('plot_/hist_doc_len.pdf')
