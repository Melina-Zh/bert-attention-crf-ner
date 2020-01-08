# coding=utf-8


class Config(object):
    def __init__(self):
        self.label_file = './data/tag.txt'
        self.train_file = './data/dataset/15semeval_rest_train_set.txt'
        self.dev_file = './data/dataset/15semeval_rest_test_set.txt'
        self.test_file = './data/test.txt'
        self.vocab = './data/bert/vocab.txt'
        self.domain_file = './model/domain_word.txt'
        self.max_length = 42
        self.use_cuda = False
        self.patience = 7
        self.acc_f = "att_base.log"
        self.gpu = 0
        self.batch_size = 64
        self.bert_path = './data/bert'
        self.bert_embedding = 768
        self.dropout1 = 0.5
        self.dropout_ratio = 0.5
        self.lr = 0.00002
        self.lr_decay = 0.00001
        self.weight_decay = 0.00005
        self.checkpoint = 'result/BERT_CRF/'
        self.optim = 'Adam'
        self.load_model = False
        self.load_path = None
        self.base_epoch = 100
        self.output_file = "output_file.log"

    def update(self, **kwargs):
        print("yunxingdao zheli")
        print(kwargs.items())
        for k, v in kwargs.items():
            print("k和v的值")
            print(k,v)
            setattr(self, k, v)

    def __str__(self):

        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])


if __name__ == '__main__':

    con = Config()
    con.update(gpu=8)
    print(con.gpu)
    print(con)
