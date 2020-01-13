"""
plot precision-recall bar chart from json dictionary
"""

import matplotlib.pyplot as plt
import json
import pickle
import numpy as np

plot_fscore = False

if plot_fscore:
	filename = '8'
	data = json.load(open('pickle/pr_' + filename + '.json', 'r'))
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
	# ax.set_xticks(c)
	# ax.set_xticklabels(c)
	ax.set_xlabel("Value of parameter C")
	plt.legend()
	plt.grid(True)

	plt.show()

else:
	# AUC-ROC values
	final_results = pickle.load(open('pickle/final_res_8' + '.pkl', 'rb'))
	cvals = list(np.arange(0, 5, 0.1))
	prec_recall_dat = {}

	for c in cvals:
	    prec_recall_dat[c] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
	    for word, res in final_results.items():
	        for iteration, d in res.items():
	            if d[0][2] == 'right':
	                found = False
	                for gr, pred, _ in d:
	                    if pred + c >= gr:
	                        found = True
	                if found:
	                    prec_recall_dat[c]['tp'] += 1
	                else:
	                    prec_recall_dat[c]['fn'] += 1
	            else:
	                found = False
	                for gr, pred, _ in d:
	                    if pred + c >= gr:
	                        found = True
	                if found:
	                    prec_recall_dat[c]['fp'] += 1
	                else:
	                    prec_recall_dat[c]['tn'] += 1

	# store metrics in dictionary
	roc = []
	for c, vals in prec_recall_dat.items():
	    fpr = vals['fp'] / (vals['fp'] + vals['tn'])
	    tpr = vals['tp'] / (vals['tp'] + vals['fn'])
	    roc.append((fpr, tpr))

	mini = 1
	eer = 1
	for fpr, tpr in roc:
		if abs(1-fpr-tpr) < mini:
			mini = abs(1-fpr-tpr)
			eer = fpr

	print("EER:", eer)
	
	plt.plot([x[0] for x in roc], [x[1] for x in roc])
	plt.grid(True)
	plt.ylabel("TPR")
	plt.xlabel("FPR")
	plt.show()
