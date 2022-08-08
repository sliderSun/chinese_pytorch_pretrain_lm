import torch
import random
from tqdm import tqdm
import fairies as fa
from transformers.models.bert import BertTokenizer
from transformers import BertConfig, BertModel, BertForSequenceClassification
from bert4keras.snippets import sequence_padding, DataGenerator
from transformers import AdamW, get_linear_schedule_with_warmup, get_scheduler
import numpy as np
from bert4keras.tokenizers import Tokenizer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(23)


maxlen = 128
batch_size = 16
epochs = 5
lr = 2e-5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 模型路径
path = r'../models/hflroberta/'
config_path = path + r'\bert_config.json'
checkpoint_path = path + r'\pytorch_model.bin'
dict_path = path + r'\vocab.txt'

def load_data(filename):
    """加载数据
    单条格式：(文本1, 文本2, 标签id)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text1, text2, label = l.strip().split('\t')
            D.append((text1, text2, int(label)))
    return D

# def load_data(filename):
#     """加载数据
#     单条格式：(文本1, 文本2, 标签id)
#     """
#     D = []
#     data = fa.read_json(filename)
#     for d in data:
#         text1, text2, label = d[0], d[1], d[2]
#         D.append((text1, text2, int(label)))
#     return D

# task_name = 'cf'
task_name = 'LCQMC'
data_path = './data/'
# datasets = [
#     load_data('%s/%s/%s_cf.json' % (data_path, task_name, f))
#     for f in ['train', 'valid']
# ]
datasets = [
    load_data('%s%s/%s.%s.data' % (data_path, task_name, task_name, f))
    for f in ['train', 'valid', 'test']
]

# 加载数据集
# train_data, valid_data = datasets
train_data, valid_data, test_data = datasets
train_data = train_data
train_data = train_data[:1000]
valid_data = valid_data
valid_data = valid_data[:500]
test_data = test_data[:500]

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_masks, batch_labels = [], [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                text1, text2, maxlen=maxlen
            )
            masks = [1] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_masks.append(masks)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_masks = sequence_padding(batch_masks)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids, batch_masks], batch_labels
                batch_token_ids, batch_segment_ids, batch_masks, batch_labels = [], [], [], []

train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
# test_generator = data_generator(test_data, batch_size)


# 模型
bert_config = BertConfig.from_pretrained(config_path)
bert_config.num_labels = 2
model = BertForSequenceClassification.from_pretrained(checkpoint_path, config=bert_config)
model = model.to(device)

# 优化器
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'LayerNorm.gamma', 'LayerNorm.beta', ]
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

# optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

criterion = torch.nn.CrossEntropyLoss()

best_accuracy = 0.
for epoch in range(epochs):
    model.train()
    totle_loss = 0.
    for batchs, batch_labels in tqdm(train_generator):
        batch_token_ids, batch_sent_ids, batch_masks = batchs
        batch_token_ids = torch.LongTensor(batch_token_ids).to(device)
        batch_sent_ids = torch.LongTensor(batch_sent_ids).to(device)
        batch_masks = torch.LongTensor(batch_masks).to(device)
        batch_labels = batch_labels[:, 0]
        batch_labels = torch.LongTensor(batch_labels).to(device)
        out = model(batch_token_ids, batch_masks, batch_sent_ids)
        logits = out.logits
        loss = criterion(logits, batch_labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        totle_loss += loss.item()
    print(totle_loss / len(train_generator))
    model.eval()
    right_num = 0
    totle_num = 0
    with torch.no_grad():
        for batchs, batch_labels in tqdm(valid_generator):
            batch_token_ids, batch_sent_ids, batch_masks = batchs
            batch_token_ids = torch.LongTensor(batch_token_ids).to(device)
            batch_sent_ids = torch.LongTensor(batch_sent_ids).to(device)
            batch_masks = torch.LongTensor(batch_masks).to(device)
            batch_labels = batch_labels[:, 0]
            batch_labels = torch.LongTensor(batch_labels).to(device)
            out = model(batch_token_ids, batch_masks, batch_sent_ids)
            logits = out.logits
            pre = torch.max(logits, dim=-1)[1]
            right_num += (pre == batch_labels).sum()
            totle_num += batch_token_ids.shape[0]
    acc = right_num * 1.0 / totle_num
    if acc > best_accuracy:
        best_accuracy = acc

    print('Epoch: {}, dev_acc={:6f}, best_acc={:6f}'.format(epoch + 1, acc, best_accuracy))




























































