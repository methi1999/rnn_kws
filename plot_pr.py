"""
plot precision-recall bar chart from json dictionary
"""

import matplotlib.pyplot as plt
import json
import numpy as np

filename = 'pr'

data = json.load(open(filename + '.json', 'r'))
print(data)

prec = [x['prec-recall'][0] for x in data.values()]
recall = [x['prec-recall'][1] for x in data.values()]
fscore = [x['prec-recall'][2] for x in data.values()]
c = np.array([float(x) for x in list(data.keys())])
print(prec, recall, fscore, c)

width = 0.03
ax = plt.subplot(111)
ax.bar(c - width, prec, width=width, color='b', align='center', label='Precision')
ax.bar(c, recall, width=width, color='g', align='center', label='Recall')
ax.bar(c + width, fscore, width=width, color='r', align='center', label='F-Score')
ax.set_xticks(c)
ax.set_xticklabels(c)
ax.set_xlabel("Value of parameter C")
plt.legend()
plt.grid(True)

plt.show()
