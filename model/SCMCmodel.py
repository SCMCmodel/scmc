!pip install fastnlp

from torch import nn
from fastNLP.modules import encoder, decoder
from fastNLP.embeddings.utils import get_embeddings
import torch
from fastNLP.modules.encoder import MaxPool
from fastNLP.models.base_model import BaseModel
from fastNLP.core._logger import logger
import warnings
from fastNLP.embeddings.bert_embedding import BertEmbedding
from sklearn.manifold import TSNE
class BibertSCV(BaseModel):
    def __init__(self, embed: BertEmbedding, num_labels: int = 2, num_tags: int = 7,  hidden_size=256,  temperature = 1,num_layers=1, dropout=0.3):
        r"""

        :param fastNLP.embeddings.BertEmbedding embed: 下游模型的编码器(encoder).
        :param int num_labels: 文本分类类别数目，默认值为2.
        :param float dropout: dropout的大小，默认值为0.1.
        """
        super(BibertSCV, self).__init__()
        self.temperature = 0.1
        self.num_labels = num_labels
        self.num_tags = num_tags
        self.bert = embed
        self.m = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=dropout)

        # self.fc_class = nn.Linear(self.bert.embedding_dim, num_labels)
        # self.fc_tag = nn.Linear(self.bert.embedding_dim, num_tags)
        self.critrion_class = nn.MultiLabelSoftMarginLoss()
        self.critrion_tag = nn.MultiLabelSoftMarginLoss()
        self.ce_class = nn.CrossEntropyLoss()
        self.ce_tag = nn.CrossEntropyLoss()

        self.lstm = encoder.LSTM(self.bert.embedding_dim, hidden_size=hidden_size // 2, num_layers=num_layers,
                                 bidirectional=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc_class = nn.Linear(hidden_size, num_labels)
        self.fc_tag = nn.Linear(hidden_size, num_tags)
        self.maxpooling = MaxPool(self.bert.embedding_dim)


#         if not self.bert.model.include_cls_sep:
#             self.bert.model.include_cls_sep = True
#             warn_msg = "Bert for sequence classification excepts BertEmbedding `include_cls_sep` True, " \
#                        "but got False. FastNLP has changed it to True."
#             logger.warning(warn_msg)
#             warnings.warn(warn_msg)
#             print(warn_msg)

    def forward(self, words, seq_len, target1, target2, flags):
        r"""
        输入为 [[w1, w2, w3, ...], [...]], BERTEmbedding会在开头和结尾额外加入[CLS]与[SEP]
        :param torch.LongTensor words: [batch_size, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.Tensor [batch_size, num_labels]
        """
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        input_x = self.bert(words)
        hidden = self.dropout(input_x)
        outputs, (_, _) = self.lstm(hidden, seq_len)
        outputs = self.dropout_layer(outputs)
        feat = self.maxpooling(outputs)

        out_class = self.fc_class(feat)
        out_tag = self.fc_tag(feat)
        if flags == 0:
            loss_class_c , loss_tag_c = self.scl(feat, seq_len, target1, target2, flags)
            loss_class = self.ce_class(out_class, target1)
            loss_tag = self.ce_tag(out_tag, target2)

            return loss_class_c , loss_tag_c, loss_class, loss_tag
        if flags == 2:
            class_pre = torch.max(out_class, -1)[1]
            tag_pre = torch.max(out_tag, -1)[1]
            
            return class_pre,tag_pre,out_class,out_tag

    def scl(self,feature, seq_len, target1, target2, flags):
        feature_x = self.m(feature)
        temp_feature =  torch.matmul(feature_x, feature.permute(1,0))

        logit = torch.divide(temp_feature, self.temperature)
        loss_class, loss_tag = self.scl_loss(logit, seq_len, target1, target2, flags)
        return loss_class, loss_tag

    def scl_loss(self,logit, seq_len, target1, target2, flags):
        class_pred = logit
        class_ture = target1.type_as(class_pred)

        tag_pred = logit
        tag_ture = target2.type_as(tag_pred)

        class_ture_x = torch.unsqueeze(class_ture,-1)
        class_ture_y = (torch.eq(class_ture_x,class_ture_x.permute(1,0))).type_as(class_pred)
        class_ture_z = class_ture_y/torch.sum(class_ture_y, 1, keepdim=True)

        tag_ture_x = torch.unsqueeze(tag_ture, -1)
        tem = tag_ture_x.permute(1, 0)
        te = torch.eq(tag_ture_x, tem)
        tag_ture_y = (te).type_as(tag_pred)
        temp = torch.sum(tag_ture_y, 1, keepdim=True)
        tag_ture_z = tag_ture_y/temp

        class_cross_entropy = self.critrion_class(class_pred, class_ture_z)
        tag_cross_entropy = self.critrion_class(tag_pred, tag_ture_z)

        return class_cross_entropy, tag_cross_entropy

    def predict(self, words, seq_len, target1, target2, flags):
        r"""
        :param torch.LongTensor words: [batch_size, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.LongTensor [batch_size]
        """
        logits = self.forward(words)[Const.OUTPUT]
        if self.num_labels > 1:
            return {Const.OUTPUT: torch.argmax(logits, dim=-1)}
        else:
            return {Const.OUTPUT: logits}
        
# This is a sample Python script.

import numpy as np
from fastNLP.io.loader import CSVLoader
from fastNLP import Vocabulary
from fastNLP.embeddings import StaticEmbedding,BertEmbedding,StackEmbedding
import matplotlib.pyplot as plt
import pandas as pd
import torch
from fastNLP import AccuracyMetric, ClassifyFPreRecMetric,ConfusionMatrixMetric
from sklearn.manifold import TSNE
from torch.optim import Adam
from fastNLP import BucketSampler
from fastNLP import DataSetIter
import random
from torch.nn.utils import clip_grad_norm_

#确保模型稳定性
def set_seed():
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    if not True and torch.cuda.is_available():
        torch.cuda.manual_seed_all(9)

if True and torch.cuda.is_available():
    device = torch.device("cuda", torch.cuda.current_device())
    use_cuda = True
else:
    device = torch.device("cpu")
    use_cuda = False


#数据预处理
def data_process():
    # 数据加载
    # 加载ID数据
    data_set_loader = CSVLoader(headers=('raw_words', 'target1','target2'), sep=',')
    data_set = data_set_loader.load('../input/20220810/addxg/')

    # 将数据格式转换成Dataset
    # ID数据
    dataset_tr = data_set.get_dataset('train')
    dataset_dev = data_set.get_dataset('dev')
    dataset_test = data_set.get_dataset('test')

    # 预处理数据
    # 将句子分成单词形式, 详见DataSet.apply()方法
    dataset_tr.apply(lambda ins: ins['raw_words'].split(), new_field_name='words')
    dataset_dev.apply(lambda ins: ins['raw_words'].split(), new_field_name='words')
    dataset_test.apply(lambda ins: ins['raw_words'].split(), new_field_name='words')

    # 将句子分成单词形式, 句子长度
    dataset_tr.apply(lambda ins: len(ins['raw_words'].split()), new_field_name='seq_len')
    dataset_dev.apply(lambda ins: len(ins['raw_words'].split()), new_field_name='seq_len')
    dataset_test.apply(lambda ins: len(ins['raw_words'].split()), new_field_name='seq_len')


    # 将target数据类型进行转换
    dataset_tr.apply(lambda ins: ins['target1'], new_field_name='target1')
    dataset_dev.apply(lambda ins: ins['target1'], new_field_name='target1')
    dataset_test.apply(lambda ins: ins['target1'], new_field_name='target1')

    dataset_tr.apply(lambda ins: ins['target2'], new_field_name='target2')
    dataset_dev.apply(lambda ins: ins['target2'], new_field_name='target2')
    dataset_test.apply(lambda ins: ins['target2'], new_field_name='target2')
    #
    dataset_tr.set_input('words', 'seq_len','target1','target2')
    dataset_tr.set_target('target1','target2')
    dataset_dev.set_input('words', 'seq_len','target1','target2')
    dataset_dev.set_target('target1','target2')
    dataset_test.set_input('words', 'seq_len','target1','target2')
    dataset_test.set_target('target1','target2')

    # 建立词表,转换为index
    vocab = Vocabulary()
    #  从该dataset中的chars列建立词表,验证集或者测试集在建立词表是放入no_create_entry_dataset这个参数中。
    vocab.from_dataset(dataset_tr, field_name='words', no_create_entry_dataset=[dataset_dev, dataset_test])

    #  使用vocabulary将chars列转换为index
    vocab.index_dataset(dataset_tr, dataset_dev, dataset_test, field_name='words')

    target1_vocab = Vocabulary(unknown=None, padding=None)
    target1_vocab.from_dataset(dataset_tr, field_name='target1', no_create_entry_dataset=[dataset_dev, dataset_test])
    #  使用vocabulary将chars列转换为index
    target1_vocab.index_dataset(dataset_tr, dataset_dev, dataset_test, field_name='target1')

    target2_vocab = Vocabulary(unknown=None, padding=None)
    target2_vocab.from_dataset(dataset_tr, field_name='target2', no_create_entry_dataset=[dataset_dev, dataset_test])
    #  使用vocabulary将chars列转换为index
    target2_vocab.index_dataset(dataset_tr, dataset_dev, dataset_test, field_name='target2')


    return dataset_tr,dataset_dev,dataset_test,vocab,target1_vocab,target2_vocab

#模型训练函数
def main(modelname):
    #加载数据
    trian_data, dev_data, test_test, wor_vocab , tar1_vocab, tar2_vocab = data_process()

    # 选择预训练词向量
    #word2vec_embed_tr = StaticEmbedding(wor_vocab, model_dir_or_name='en-glove-6b-200d')
    word2vec_embed_tr = BertEmbedding(wor_vocab, model_dir_or_name='en-base-cased', requires_grad=False,include_cls_sep=True)

    #实例化模型
    model = BibertSCV(word2vec_embed_tr, len(tar1_vocab), len(tar2_vocab))

    # 确定模型采用的训练GPU CPU
    use_gpu = True
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
        use_cuda = True
    else:
        device = torch.device("cpu")
        use_cuda = False
    if use_cuda:
        model.cuda(device)

    # 模型优化算法
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)

    best_class_acc = [0.0, 0.0, 0.0]
    best_tag_acc = [0.0, 0.0, 0.0]
    best_total_acc = [0.0, 0.0, 0.0]

    # 读取数据集二次封装
    batch_size =16
    train_sampler = BucketSampler(batch_size=batch_size, seq_len_field_name='seq_len')
    train_batch = DataSetIter(batch_size=batch_size, dataset=trian_data, sampler=train_sampler)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20,40], gamma=0.5, last_epoch=-1)
    # 开始训练

    for epoc in range(60):
        step = 0
        loss = 0

        for b_words, b_target in train_batch:
            step += 1
            model.zero_grad()

            # 将数据加载到GPU上
            if use_cuda:
                target1, target2, words, seq_len = \
                    b_words['target1'].cuda(), b_words['target2'].cuda(), \
                    b_words['words'].cuda(), b_words['seq_len'].cuda()
            else:
                target1, target2, words, seq_len = \
                    b_words['target1'], b_words['target2'], \
                    b_words['words'], b_words['seq_len']

            # 算出loss，并进行处理
            flag = 0
            loss_class_c , loss_tag_c,class_loss, tag_loss = model.forward(words, seq_len, target1, target2, flag)
            
            
                
            loss = 2*(loss_class_c * class_loss)/(loss_class_c + class_loss) + 2*loss_tag_c*tag_loss/(loss_tag_c+tag_loss)
                
           


            loss.backward()

            clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            if step % 100 == 0:
                print("loss_csv domain:", loss.item())
                print('epoch: {}|    step: {} |    loss: {}'.format(epoc, step, loss.item()))
        # 对模型进行测试dev数据集
        # class_acc, tag_acc, total_acc = dev(model, dev_data, tar1_vocab, tar2_vocab)
    

        class_acc, tag_acc, total_acc = dev(model, dev_data, tar1_vocab, tar2_vocab)

        if class_acc > best_class_acc[0]:
            best_class_acc = [class_acc, tag_acc, total_acc, epoc]
            torch.save(model, '/kaggle/working/' + modelname + '_cla.bin')
        if tag_acc > best_tag_acc[1]:
            torch.save(model, '/kaggle/working/' + modelname + '_tag.bin')
            best_tag_acc = [class_acc, tag_acc, total_acc, epoc]
        if total_acc > best_total_acc[2]:
            torch.save(model, '/kaggle/working/' + modelname + '_tot.bin')
            best_total_acc = [class_acc, tag_acc, total_acc, epoc]

        scheduler.step()
        
        
        del loss
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    print("best_class_acc:", best_class_acc)
    print("best_tag_acc:", best_tag_acc)
    print("best_total_acc:", best_total_acc)

def dev(model, dev_data,tar1_vocab, tar2_vocab):
    model.eval()
    use_cuda = True
    eval_loss_class = 0
    eval_loss_tag = 0
    pred_classes= []
    true_classes = []
    pred_tags = []
    true_tags= []

    # batch_size = 50  #atis
    batch_size = 16
    dev_sampler = BucketSampler(batch_size=batch_size, seq_len_field_name='seq_len')
    dev_batch = DataSetIter(batch_size=batch_size, dataset=dev_data, sampler=dev_sampler)
    with torch.no_grad():
        for temp_data,temp_x in dev_batch:
            if use_cuda:
                target1, target2, words, seq_len = \
                    temp_data['target1'].cuda(), temp_data['target2'].cuda(), \
                    temp_data['words'].cuda(), temp_data['seq_len'].cuda()
            else:
                target1, target2, words, seq_len = \
                    temp_data['target1'], temp_data['target2'], \
                    temp_data['words'], temp_data['seq_len']
            flag =0
            loss_class_c , loss_tag_c, class_loss, tag_loss = model.forward(words, seq_len, target1, target2,flag)
            flag = 2
            pred_class,pred_tag,out_class,out_tag = model.forward(words,seq_len,target1,target2,flag)

            pred_classes.extend(pred_class.cpu().numpy().tolist())
            true_classes.extend(target1.cpu().numpy().tolist())
            pred_tags.extend(pred_tag.cpu().numpy().tolist())
            true_tags.extend(target2.cpu().numpy().tolist())
            eval_loss_class += class_loss.item()
            eval_loss_tag += tag_loss.item()

    metric_class = AccuracyMetric()
    metric_class.evaluate(torch.tensor(pred_classes, dtype=torch.float32),torch.tensor(true_classes, dtype=torch.float32))
    rel_class = metric_class.get_metric()

    metric_tag = AccuracyMetric()
    metric_tag.evaluate(torch.tensor(pred_tags, dtype=torch.float32),torch.tensor(true_tags, dtype=torch.float32))
    rel_tag = metric_tag.get_metric()
    
    avg_loss_class = eval_loss_class*batch_size/len(dev_data)
    avg_loss_tag = eval_loss_tag*batch_size/len(dev_data)

    acc_total = total_acc(pred_classes, true_classes, pred_tags, true_tags)
    print('\nDEV-Evaluation - class_loss: {:.4f}  class_acc: {:.4f} tag_loss: {:.4f} '
          'tag_acc: {:.4f} total_acc: {:.4f} \n'.format(avg_loss_class,rel_class['acc'],avg_loss_tag,rel_tag['acc'],acc_total))
    model.train()
    return rel_class['acc'], rel_tag['acc'], acc_total

def test(modelname):
    #加载数据
    trian_data, dev_data, test_test, wor_vocab, tar1_vocab, tar2_vocab = data_process()


    model = torch.load(modelname,map_location=device)
    model.eval()
    pred_classes = []
    true_classes = []
    pred_tags = []
    true_tags = []
    out_tages = []
    out_classes = []

    # batch_size = 893
    batch_size = 16
    test_sampler = BucketSampler(batch_size=batch_size, seq_len_field_name='seq_len')
    test_batch = DataSetIter(batch_size=batch_size, dataset=test_test, sampler=test_sampler)

    for temp_data, temp_x in test_batch:

        if use_cuda:
            target1, target2, words, seq_len = \
                temp_data['target1'].cuda(), temp_data['target2'].cuda(), \
                temp_data['words'].cuda(), temp_data['seq_len'].cuda()
        else:
            target1, target2, words, seq_len = \
                temp_data['target1'], temp_data['target2'], \
                temp_data['words'], temp_data['seq_len']
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        flag = 2
        pred_class, pred_tag,out_class,out_tag = model.forward(words, seq_len, target1, target2, flag)

        pred_classes.extend(pred_class.cpu().numpy().tolist())
        true_classes.extend(target1.cpu().numpy().tolist())
        pred_tags.extend(pred_tag.cpu().numpy().tolist())
        true_tags.extend(target2.cpu().numpy().tolist())
        out_classes.extend(out_class.cpu().detach().numpy().tolist())
        out_tages.extend(out_tag.cpu().detach().numpy().tolist())
    metric_class = AccuracyMetric()
    metric_class.evaluate(torch.tensor(pred_classes, dtype=torch.float32),torch.tensor(true_classes, dtype=torch.float32))
    rel_class = metric_class.get_metric()

    fpr_class = ClassifyFPreRecMetric(f_type='macro')
    fpr_class.evaluate(torch.tensor(pred_classes, dtype=torch.float32),torch.tensor(true_classes, dtype=torch.float32))
    nfpr_class = fpr_class.get_metric()

    confu_class = ConfusionMatrixMetric()
    confu_class.evaluate(torch.tensor(pred_classes, dtype=torch.float32),torch.tensor(true_classes, dtype=torch.float32))
    ncf_class = confu_class.get_metric()

    metric_tag = AccuracyMetric()
    metric_tag.evaluate(torch.tensor(pred_tags, dtype=torch.float32),torch.tensor(true_tags, dtype=torch.float32))
    rel_tag = metric_tag.get_metric()

    fpr_tag = ClassifyFPreRecMetric(f_type='macro')
    fpr_tag.evaluate(torch.tensor(pred_tags, dtype=torch.float32),torch.tensor(true_tags, dtype=torch.float32))
    nfpr_tag = fpr_tag.get_metric()

    confu_tag = ConfusionMatrixMetric()
    confu_tag.evaluate(torch.tensor(pred_tags, dtype=torch.float32),torch.tensor(true_tags, dtype=torch.float32))
    ncf_tag = confu_tag.get_metric()

    acc_total = total_acc(pred_classes, true_classes, pred_tags, true_tags)
  
    print('**************type**********************')
    print('\nEvaluation -  class_acc: {:.4f}  f1: {:.4f} pre: {:.4f} rec: {:.4f}\n'.format(rel_class['acc'], nfpr_class['f'], nfpr_class['pre'], nfpr_class['rec']))
    print(ncf_class)
    print('****************event********************')
    print('\nEvaluation -  tag_acc: {:.4f}   f1: {:.4f}  pre: {:.4f}   rec: {:.4f} \n'.format(rel_tag['acc'], nfpr_tag['f'], nfpr_tag['pre'], nfpr_tag['rec']))
    print(ncf_tag)
    print('****************total_acc********************')
    print('\nEvaluation -  acc_total: {:.4f}  \n'.format(acc_total))
    visualize_embeddings(out_classes,true_classes,2)
    visualize_embeddings(out_tages,true_tags,8)

def total_acc(pred_classes, real_classes, pred_tags, real_tags):
    #计算sent_acc
    total_count, correct_count = 0.0, 0.0
    for p_class, r_class, p_tag, r_tag in zip(pred_classes, real_classes, pred_tags, real_tags):
        if p_class == r_class and p_tag == r_tag:
            correct_count += 1.0
        total_count += 1.0
    return 1.0 * correct_count / total_count

def visualize_embeddings(embeddings, labels,num, figsize=(16, 16)):
    # Extract TSNE values from embeddings
    embed2D = TSNE(n_components=2, n_jobs=-1, random_state=0).fit_transform(embeddings)
    embed2D_x = embed2D[:, 0]
    embed2D_y = embed2D[:, 1]

    # Create dataframe with labels and TSNE values
    df_embed = pd.DataFrame({'labels': labels})
    df_embed = df_embed.assign(x=embed2D_x, y=embed2D_y)

    # Create classes dataframes
    if num == 2:
        df_embed_cbb = df_embed[df_embed['labels'] == 0]
        df_embed_cbsd = df_embed[df_embed['labels'] == 1]
        

        # Plot embeddings
        plt.figure(figsize=figsize)
        plt.scatter(df_embed_cbb['x'], df_embed_cbb['y'], color='yellow', s=10, label='informative')
        plt.scatter(df_embed_cbsd['x'], df_embed_cbsd['y'], color='blue', s=10, label='not_informative')
        
        
    if num == 8:
        df_embed_cbb = df_embed[df_embed['labels'] == 0]
        df_embed_cbsd = df_embed[df_embed['labels'] == 1]
        df_embed_cgm = df_embed[df_embed['labels'] == 2]
        df_embed_cmd = df_embed[df_embed['labels'] == 3]
        df_embed_c = df_embed[df_embed['labels'] == 4]
        df_embed_cb = df_embed[df_embed['labels'] == 5]
        df_embed_cg = df_embed[df_embed['labels'] == 6]
        df_embed_cm = df_embed[df_embed['labels'] == 7]
       
        # Plot embeddings{0: 'rescue donation effort', 1: 'sympathy and support', 2: 'injured dead people', 3: 'other relevant information', 4: 'infrastructure utility damage', 
        #{0: 'other_relevant_information', 1: 'not_humanitarian', 2: 'rescue_volunteering_or_donation_effort', 3: 'infrastructure_and_utility_damage', 4: 'injured_or_dead_people', 5: 'affected_individuals', 6: 'vehicle_damage', 7: 'missing_or_found_people'}
        plt.figure(figsize=figsize)
        plt.scatter(df_embed_cbb['x'], df_embed_cbb['y'], color='yellow', s=10, label='other_relevant_information')
        plt.scatter(df_embed_cbsd['x'], df_embed_cbsd['y'], color='blue', s=10, label='not_humanitarian')
        plt.scatter(df_embed_cgm['x'], df_embed_cgm['y'], color='red', s=10, label='rescue_volunteering_or_donation_effort')
        plt.scatter(df_embed_cmd['x'], df_embed_cmd['y'], color='orange', s=10, label='infrastructure_and_utility_damage')
        plt.scatter(df_embed_c['x'], df_embed_c['y'], color='black', s=10, label='injured_or_dead_people')
        plt.scatter(df_embed_cb['x'], df_embed_cb['y'], color='grey', s=10, label='affected_individuals')
        plt.scatter(df_embed_cg['x'], df_embed_cg['y'], color='purple', s=10, label='vehicle_damage')
        plt.scatter(df_embed_cm['x'], df_embed_cm['y'], color='green', s=10, label='missing_or_found_people')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    set_seed()
    main('snips_3')
    if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    test('/kaggle/working/snips_3_cla.bin')
    test('/kaggle/working/snips_3_tag.bin')
    test('/kaggle/working/snips_3_tot.bin')