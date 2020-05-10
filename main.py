# coding=utf-8
# import warnings
# import traceback
# import sys

# def warn_with_traceback(message, category, filename, lineno, fil=None, line=None):
#     log = file if hasattr(file, 'write') else sys.stderr
#     trackback.print_stack(file=log)
#     log.write(warnings.formatwarning(message, category, filename, lineno, oine))

# warnings.showwarning = warn_with_traceback
# warnings.simplefilter("error")

import torch
import torch.nn as nn
from torch.autograd import Variable
from config import Config
import torch.optim as optim
from utils import load_vocab, read_corpus, load_model, save_model, EarlyStopping
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import fire
import logging
from pytorch_pretrained_bert.tokenization import BertTokenizer
from model.bert_attention_crf import BERT_CRF
import os
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import numpy as np
import time
from tqdm import tqdm, trange
logger = logging.getLogger(__name__)
import conlleval

def readfile(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    f = open(filename)
    data = []
    sentence = []
    label= []
    for line in f:
        line = line.strip()
        splits = line.split('|||')
        sentence = splits[0].split(' ')
        label = splits[1].split(' ')
        data.append((sentence, label))

    return data


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               mask,
               label_ids,
               is_real_example=True):
    self.input_ids = input_ids
    self.mask = mask
    #self.segment_ids = segment_ids
    self.label_ids = label_ids
    self.is_real_example = is_real_example

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 0)}

    features = []
    all_tokens=[]
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []

        label_mask = []
        for i, (word, label) in enumerate(zip(textlist, labellist)):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            for i, _ in enumerate(token):
                if i == 0:
                    labels.append(label)
                else:
                    labels.append("X")
        if len(tokens) >= max_seq_length - 2:
            tokens = tokens[0:(max_seq_length - 2)] # [CLS] and [SEP]
            labels = labels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        mask = [True] * len(input_ids)
        # use zero to padding and you should
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            mask.append(False)
            segment_ids.append(0)
            label_ids.append(0)
            ntokens.append("[PAD]")
        assert len(input_ids) == max_seq_length
        assert len(mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(ntokens) == max_seq_length

        if ex_index < 3:
            logging.info("*** Example ***")
            logging.info("guid: %s" % (example.guid))
            logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logging.info("input_mask: %s" % " ".join([str(x) for x in mask]))
            logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                mask=mask,
                label_ids=label_ids,
            )
        )
        all_tokens.append(ntokens)
    return features, all_tokens

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class NerProcessor():
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(data_dir), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return ["O", "B-AP", "I-AP", "X", "[CLS]", "[SEP]"]

    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def result_to_pair(writer, dev_features, result, label_list, max_length):
    idx = 0

    for each_feature in dev_features:
        i = 0
        line = ''
        line_token = each_feature.input_ids
        label_token = each_feature.label_ids
        len_seq = len(label_token)

        if len(line_token) != len(label_token):
            logger.info(each_feature.input_ids)
            logger.info(each_feature.label_ids)
            break
        while label_token[i] != 5: # [SEP]

            if label_token[i] == 3 or label_token[i] == 4: # X [CLS]
                i += 1
                continue
            if result[idx][i] == 3 or result[idx][i] == 4 or result[idx][i] == 5:
                result[idx][i] = 0
            curr_labels = label_list[result[idx][i]]

            try:
                line += str(line_token[i]) + ' ' + label_list[label_token[i]] + ' ' + curr_labels + '\n'

            except Exception as e:
                logger.info(e)
                logger.info(each_feature.input_ids)
                logger.info(each_feature.label_ids)
                line = ''
                break
            i += 1



        writer.write(line)
        idx += 1

def train(**kwargs):
    config = Config()
    config.update(**kwargs)
    print('当前设置为:\n', config)
    if config.use_cuda:
        torch.cuda.set_device(config.gpu)
    print('loading corpus')
    vocab = load_vocab(config.vocab)
    label_dic = load_vocab(config.label_file)
    early_stopping = EarlyStopping(patience=config.patience, verbose=True)
    tagset_size = len(label_dic)
    '''
    train_data = read_corpus(config.train_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab)
    dev_data = read_corpus(config.dev_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab)
    
    train_ids = torch.LongTensor([temp.input_id for temp in train_data])
    train_masks = torch.LongTensor([temp.input_mask for temp in train_data])
    train_tags = torch.LongTensor([temp.label_id for temp in train_data])

    train_dataset = TensorDataset(train_ids, train_masks, train_tags)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)
    '''

    model = BERT_CRF(config.bert_path, tagset_size, config.bert_embedding, config.bert_embedding, dropout_ratio=config.dropout_ratio, dropout1=config.dropout1, use_cuda=config.use_cuda)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    processor = NerProcessor()
    device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1
    train_examples = processor.get_train_examples(config.train_file)

    train_features, train_all_tokens = convert_examples_to_features(
        train_examples, label_list, config.max_length, tokenizer)

    dev_examples = processor.get_train_examples(config.dev_file)

    dev_features, dev_all_tokens = convert_examples_to_features(
        dev_examples, label_list, config.max_length, tokenizer)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", config.batch_size)

    dev_ids = torch.LongTensor([temp.input_ids for temp in dev_features])
    dev_masks = torch.BoolTensor([temp.mask for temp in dev_features])
    dev_tags = torch.LongTensor([temp.label_ids for temp in dev_features])

    dev_dataset = TensorDataset(dev_ids, dev_masks, dev_tags)
    dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=config.batch_size)

    '''
    domain part
    
    domain_f = open(config.domain_file, "r")
    domain_list = []
    for i in domain_f:
        domain_list.append(i)
    domain_no_sep = tokenizer.convert_tokens_to_ids(domain_list)
    domain_no_sep = torch.LongTensor(domain_no_sep)
    '''

    all_input_ids = torch.LongTensor([f.input_ids for f in train_features])
    all_input_mask = torch.BoolTensor([f.mask for f in train_features])
    all_label_ids = torch.LongTensor([f.label_ids for f in train_features])

    train_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config.batch_size)
    time1 = time.time()
    if config.load_model:
        assert config.load_path is not None
        model = load_model(model, name=config.load_path)
    if config.use_cuda:
        model.cuda()
    model.train()
    # different lr
    crf_params = list(map(id, model.crf.parameters()))
    base_params = filter(lambda p: id(p) not in crf_params,
                         model.parameters())
    optimizer = getattr(optim, config.optim)
    optimizer = optimizer([
            {'params': base_params},
            {'params': model.crf.parameters(), 'lr': config.lr * 1.6}], lr=config.lr, weight_decay=config.weight_decay)
    eval_loss = 10000
    print(model.state_dict().keys())

    for epoch in range(config.base_epoch):
        model.train()
        step = 0
        acc_f = open(config.checkpoint+"acc.log", 'a')
        for i, batch in enumerate(tqdm(train_dataloader, desc="Epoch {} ".format(epoch))):
            batch = tuple(t.to(device) for t in batch)
            step += 1
            model.zero_grad()
            inputs, masks, tags = batch
            inputs, masks, tags = Variable(inputs), Variable(masks), Variable(tags)
            #domain_id = torch.LongTensor(domain_no_sep).view(1, len(domain_no_sep))
            if config.use_cuda:
                inputs, masks, tags = inputs.cuda(), masks.cuda(), tags.cuda()
             #   domain_id = domain_id.cuda()
            feats = model(inputs, masks)
         
            '''
            masks[tags == label_list.index('<start>')] = 0
            masks[tags == label_list.index('<eos>')] = 0
            tags[tags == label_list.index('<start>')] = label_dic["<pad>"]
            tags[tags == label_list.index('<eos>')] = label_dic["<pad>"]
            '''
            loss = model.loss(feats, masks, tags)
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                print('step: {} |  epoch: {}|  loss: {}'.format(step, epoch, loss.item()))
        acc_f.write("Epoch {} :".format(epoch))
        stop = dev(model, dev_loader, epoch, config, acc_f, early_stopping, dev_features, label_list)
        if stop:
            break

    time2 = time.time()
    #model = load_model(model, path=config.checkpoint, name=config.load_path)

    #five_r = open(config.checkpoint+"five_res.log", 'a')
    #five_r.write(''.join(test(model, dev_loader, config, dev_examples, label_list)))
    #five_r.close()
    print("total time: {:.1f}s".format(time2-time1))
    acc_f.close()


def dev(model, dev_loader, epoch, config, acc_f, early_stopping, dev_features, label_list):
    model.eval()
    eval_loss = 0
    true = []
    pred = []
    length = 0
    stop = 0
    for i, batch in enumerate(dev_loader):
        inputs, masks, tags = batch
        length += inputs.size(0)
        inputs, masks, tags = Variable(inputs), Variable(masks), Variable(tags)

        if config.use_cuda:
            inputs, masks, tags = inputs.cuda(), masks.cuda(), tags.cuda()

        feats = model(inputs, masks)
        '''
        masks[tags == label_list.index('<start>')] = 0
        masks[tags == label_list.index('<eos>')] = 0
        tags[tags == label_list.index('<start>')] = label_dic["<pad>"]
        tags[tags == label_list.index('<eos>')] = label_dic["<pad>"]
        '''
        path_score, best_path = model.crf(feats, masks.byte())
        loss = model.loss(feats, masks, tags)
        eval_loss += loss.item()
        pred.extend([t for t in best_path])
        true.extend([t for t in tags])

    with open(config.output_file, "w") as writer:
        result_to_pair(writer, dev_features, pred, label_list, config.max_length)

    early_stopping(eval_loss, model, epoch)

    if early_stopping.early_stop:
        print("Early stopping")
        stop = 1
    eval_result = conlleval.return_report(config.output_file)
    print(''.join(eval_result))
    acc_f.write(''.join(eval_result))
    acc_f.write("\n")
    acc_f.close()

    save_model(model, epoch, early_stopping.best_epoch, path=config.checkpoint)
    model.train()

    return stop


def test(model, dev_loader, config, dev_examples, label_list):
    model.eval()
    eval_loss = 0
    true = []
    pred = []
    length = 0
    stop = 0
    for i, batch in enumerate(dev_loader):
        inputs, masks, tags = batch
        length += inputs.size(0)
        inputs, masks, tags = Variable(inputs), Variable(masks), Variable(tags)

        if config.use_cuda:
            inputs, masks, tags = inputs.cuda(), masks.cuda(), tags.cuda()

        feats = model(inputs, masks)
        '''
        masks[tags == label_list.index('<start>')] = 0
        masks[tags == label_list.index('<eos>')] = 0
        tags[tags == label_list.index('<start>')] = label_dic["<pad>"]
        tags[tags == label_list.index('<eos>')] = label_dic["<pad>"]
        '''
        path_score, best_path = model.crf(feats, masks.byte())
        loss = model.loss(feats, masks, tags)
        eval_loss += loss.item()
        pred.extend([t for t in best_path])
        true.extend([t for t in tags])

    with open(config.output_file, "w") as writer:
        result_to_pair(writer, dev_examples, pred, label_list, config.max_length)

    eval_result = conlleval.return_report(config.output_file)
    print(''.join(eval_result))
    return eval_result


if __name__ == '__main__':
    fire.Fire()











