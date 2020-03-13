import numpy as np
import matplotlib.pyplot as plt
import pickle
from read_yaml import read_yaml

import json
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

db_path = 'pickle/cnn.pkl'
config = read_yaml()
np.random.seed(7)

# Load mapping
with open(config['dir']['dataset'] + 'lstm_mapping.json', 'r') as f:
    phone_to_id = json.load(f)
    # drop weights
    phone_to_id = {k: v[0] for k, v in phone_to_id.items() if k != 'PAD'}
    print("Total phone classes:", len(phone_to_id))

with open(db_path, 'rb') as f:
    out_rnn, cases = pickle.load(f)
    cases = {k: v['templates'] for k, v in cases.items()}
    # out_rnn elements are: pred_phones, node_prob
    print("Read outputs for CNN training")

blank_id = len(phone_to_id)
use_cuda = torch.cuda.is_available()
max_l = 20
batch_size = 100
num_epochs = 20
min_phones = 3
threshold = 0.7

for word, temps in cases.items():
    cases[word] = [x for x in temps if len(x) >= min_phones]

print("Templates:", cases, '\n')


def show_image(img):

    f, axarr = plt.subplots(2, 1)
    axarr[0].imshow(img[0])
    axarr[1].imshow(img[1])
    plt.show()


def out_to_img(template, out_ids, out_prob, max_l, blank_id):

    temp_img = np.zeros((len(phone_to_id) + 1, max_l))
    for i, ph in enumerate(template):
        temp_img[phone_to_id[ph], i] = 1
    for i in range(len(template), max_l):
        temp_img[blank_id, i] = 1

    out_img = np.zeros((len(phone_to_id) + 1, max_l))
    for i, ph in enumerate(out_ids):
        out_img[ph, i] = out_prob[i]
    for i in range(len(out_ids), max_l):
        out_img[blank_id, i] = 1

    final = np.stack((temp_img, out_img), axis=0)
    return final


# print(out_to_img(['aa', 'aa', 'aa'], [4,5,6,7], [1,1,1,1], 10))


def build_db(out_rnn, templates, blank_id, max_l):

    actual_max_l = 0
    final_imgs, final_label = [], []
    kwords = [x for x in templates.keys() if x != "NONE"]
    for word_present, list_data in out_rnn.items():
        for (pred_phones, node_prob, present) in list_data:

            if len(pred_phones) < min_phones:
                continue
            if len(pred_phones) > actual_max_l:
                actual_max_l = len(pred_phones)

            if word_present == 'NONE':
                temp_word = np.random.choice(kwords)
                template = templates[temp_word][np.random.randint(low=0, high=len(templates[temp_word]))]
                img = out_to_img(template, pred_phones, node_prob, max_l, blank_id)

                if len(template) > actual_max_l:
                    actual_max_l = len(template)

                final_imgs.append(img)
                final_label.append(0)
            else:
                for template in templates[word_present]:
                    if len(template) > actual_max_l:
                        actual_max_l = len(template)
                    img = out_to_img(template, pred_phones, node_prob, max_l, blank_id)
                    final_imgs.append(img)
                    final_label.append(present)

    print("Total examples:", len(final_label))
    print("Fraction of +ve examples:", sum(final_label) / len(final_label))
    print("Max length of template/output:", actual_max_l)
    return final_imgs, final_label


def train_test_batch(imgs, labels, split, batch_size):
    train_num = int(split * len(imgs))

    zipped = list(zip(imgs, labels))
    np.random.shuffle(zipped)
    imgs, labels = [x[0] for x in zipped], [x[1] for x in zipped]

    train_egs, train_labels = imgs[:train_num], labels[:train_num]
    test_egs, test_labels = imgs[train_num:], labels[train_num:]

    final_train_batches = []
    final_test_batches = []

    i = 0
    while i < len(train_egs):
        final_train_batches.append((train_egs[i:i + batch_size], train_labels[i:i + batch_size]))
        i += batch_size
    i = 0
    while i < len(test_egs):
        final_test_batches.append((test_egs[i:i + batch_size], test_labels[i:i + batch_size]))
        i += batch_size

    return final_train_batches, final_test_batches


class CNN(nn.Module):
    def __init__(self, num_phones, max_l, batch_size):
        super(CNN, self).__init__()

        final_kernels = 4
        self.batch_size = batch_size
        num_classes = num_phones + 1

        self.flatten_dim = max_l * final_kernels
        self.conv1 = nn.Conv2d(2, 16, (num_classes, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(16, final_kernels, (1, 3), padding=(0, 1))
        self.fc1 = nn.Linear(self.flatten_dim, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.flatten_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = CNN(len(phone_to_id), max_l, batch_size)
if use_cuda:
    model = model.cuda()

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
sigmoid = nn.Sigmoid()

final_imgs, final_labels = build_db(out_rnn, cases, blank_id, max_l)
train_batch, test_batch = train_test_batch(final_imgs, final_labels, 0.7, 100)


def train():

    print('Starting Training')
    model.train()

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_batch):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # show_image(inputs[0])
            inputs, labels, = torch.from_numpy(np.array(inputs)).float(), torch.from_numpy(np.array(labels)).float()[:, None]
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = sigmoid(model(inputs))
            # print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        print("Epoch: %d ; Train Loss: %f" % (epoch, running_loss/len(train_batch)))

    print('Finished Training')


def test():

    print('Testing...')
    model.eval()
    results = {'tp':0, 'fp':0, 'tn':0, 'fn':0}

    running_loss = 0.0
    for i, data in enumerate(test_batch):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # show_image(inputs[0])
        inputs, labels, = torch.from_numpy(np.array(inputs)).float(), torch.from_numpy(np.array(labels)).float()[:,
                                                                      None]
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        outputs = sigmoid(model(inputs))

        loss = criterion(outputs, labels)
        outputs, labels = outputs.detach().numpy(), labels.detach().numpy()
        for i, out in enumerate(outputs):
            if out >= threshold and labels[i] == 1:
                results['tp'] += 1
            elif out >= threshold and labels[i] == 0:
                results['fp'] += 1
            elif out < threshold and labels[i] == 0:
                results['tn'] += 1
            else:
                results['fn'] += 1

        # print statistics
        running_loss += loss.item()

    print("Test Loss: %f" % (running_loss/len(test_batch)))
    print(results)
    print('Finished testing')

train()
test()