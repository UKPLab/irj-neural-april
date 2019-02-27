import sys

sys.path.insert(0, '../')

from algorithms._summarizer import Summarizer
from nltk.tokenize import word_tokenize
from algorithms.base import Sentence
from utils.data_helpers import extract_ngrams2, prune_ngrams, untokenize
import utils.data_helpers as util
from utils.corpus_reader import CorpusReader
from pretrain.state_type import State
from rouge.rouge import Rouge
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from utils.writer import write_to_file
from pretrain.state_type import StateLengthComputer
from utils.summary_samples_reader import *

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import random
import os
import copy

import torch
from torch.autograd import Variable


class DeepTDAgent(Summarizer):
    def __init__(self, language, data_set, train_round, topic, models,
                 alpha=0.001, opt_type='adam', temperature=10.0,
                 temp_decay=1., sample_max_num=1000, interval=100,
                 base_length=200, block_num=1):

        # hyper parameters
        self.gamma = 1
        self.epsilon = 1
        self.alpha = alpha
        self.lamb = 1.0
        if data_set == 'DUC2004':
            self.sum_token_length = 115
        else:
            self.sum_token_length = 100
        self.opt_type = opt_type

        #for state length
        self.state_block_num = block_num
        self.base_length = base_length

        # training options
        self.train_round = train_round
        self.language = language
        self.state_type = 'ngram'
        self.topic = topic
        self.data_set = data_set
        self.sample_max_num = sample_max_num

        # class-level variables; used throughout each instance of the learning agent
        self.sentences = []
        self.top_ngrams_list = []
        self.stemmed_sentences_list = []
        self.stemmed_docs_list = []
        self.sim_scores = {}
        self.models = models

        # stemmers and stop words list
        self.stemmer = PorterStemmer()
        self.stoplist = set(stopwords.words(self.language))

        # some options for the sentence-to-tokens function
        self.without_stopwords = True
        self.stem = True
        self.temperature = temperature
        self.temp_decay = temp_decay
        self.loaded_flag = False
        self.interval = interval

    def load(self,docs):
        self.docs = docs
        self.load_data()

        # state_type is the states used in the Japan version; TopNGrams are used in REAPER
        if self.state_type == 'tfidf':
            self.getTopTfidfTokens()
        else:
            self.getTopGrams()

        self.loaded_flag = True


    def __call__(self, docs):
        ### load data
        if not self.loaded_flag:
            self.load(docs)

        pretrain_model_dir = os.path.join('/home/gao/PycharmProjects/DeepTD/learnt_network/',self.data_set,self.topic)
        wanted_setting = 'rewardOPTIMAL_round5000_alpha0.001_strict5'
        summary_list = []
        model_name_list = []

        for ff in os.listdir(pretrain_model_dir):
            if ff[0] == '.' or wanted_setting not in ff:
                continue
            model_name = self.getModelName(ff)
            model_name_list.append(model_name)
            hidden_layer_width = self.vec_length
            self.deep_model = torch.nn.Sequential(
                torch.nn.Linear(self.vec_length, hidden_layer_width),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_layer_width, hidden_layer_width),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_layer_width, 1),
                )
            ### read pre-trained model
            self.deep_model.load_state_dict(torch.load(os.path.join(pretrain_model_dir,ff)))
            model_idx = self.getModelIdx(model_name)
            summary_list.append(self.trainModel(model_idx))

        summary_list.append(model_name_list)
        return summary_list

    def getModelName(self, fname):
        return fname.split('_')[0][5:]

    def getModelIdx(self, mname):
        idx = 0
        for model in self.models:
            if model[0].split('/')[-1] == mname:
                return idx
            else:
                idx += 1
        print('ERROR! Model name {} not found!'.format(mname))
        return -1


    def getRankRewards(self, value_list):
        sorted_value_list = sorted(value_list)
        new_value_list = []

        for vv in value_list:
            new_value_list.append(10*(np.searchsorted(sorted_value_list,vv)+1.0)/len(sorted_value_list))

        return new_value_list

    def trainModel(self,model_idx):
        alpha = self.alpha
        temperature = self.temperature
        loss_list = []
        sample_base = []

        if self.opt_type == 'adam':
            self.optimiser = torch.optim.Adam(self.deep_model.parameters(), lr=alpha)
        else:
            self.optimiser = torch.optim.RMSprop(self.deep_model.parameters(), lr=alpha)

        hidden_layer_width = self.vec_length
        last_deep_model = torch.nn.Sequential(
            torch.nn.Linear(self.vec_length, hidden_layer_width),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_width, hidden_layer_width),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_width, 1),
        )

        # train for train_round of episodes
        for round_cnt in range(int(self.train_round)):
            if round_cnt%self.interval == 0:
                sample_base = self.updateSampleBase(sample_base,self.deep_model,model_idx,temperature)
                temperature = temperature*self.temp_decay
                last_model_dict = self.deep_model.state_dict()
                last_deep_model.load_state_dict(last_model_dict)
                #last_deep_model = copy.deepcopy(self.deep_model)

            sample_item = random.choice(sample_base)
            loss = self.deepTrain(sample_item[0],sample_item[1],last_deep_model)
            loss_list.append(loss)

        summary = self.produceSummary(self.deep_model,model_idx)
        #self.writeLearntNetwork()
        return summary

    def updateSampleBase(self,sample_base,deep_model,model_idx,temp):
        update_num = int(self.train_round/self.interval)
        add_num = int(2*self.sample_max_num/update_num)
        print('start to add {} samples to sample base'.format(add_num))

        for ii in range(add_num):
            if len(sample_base) > self.sample_max_num:
                sample_base.pop(0)
            vec, reward = self.generateNewSample(deep_model,model_idx,temp)
            new_sample = [vec,reward]
            sample_base.append(new_sample)

        print('{} new samples added to sample base; sample base size {}'.format(add_num,len(sample_base)))

        return sample_base


    def generateNewSample(self,deep_model,model_idx,temp):
        state = State(self.sum_token_length, self.base_length, len(self.sentences), self.state_block_num,
                      self.language)
        vector_list = []

        while state.available_sents != [0]:
            next_state_matrix = []
            for legal_act in state.available_sents:
                if legal_act == 0:
                    continue
                else:
                    vector = state.getNewStateVec(legal_act-1, self.top_ngrams_list, self.sentences, self.state_type)
                    next_state_matrix.append(vector)

            ### get the prob. of each proceeding state
            value_list = self.getValueList(next_state_matrix,deep_model,temp)
            idx = self.acceptByChance(value_list)
            vector_list.append(next_state_matrix[idx])
            act = state.available_sents[idx + 1] - 1
            state.updateState(act, self.sentences)

        return vector_list, state.getOptimalTerminalReward(self.models[model_idx])

    def getValueList(self, vec_list, deep_model, temperature):
        value_variables = deep_model(Variable(torch.from_numpy(np.array(vec_list)).float()))
        value_list = value_variables.data.numpy()
        value_list /= temperature

        max_value = max(value_list)
        sum = 0.0
        nom_list = []
        for value in value_list:
            nom = np.exp((value - max_value))
            nom_list.append(nom)
            sum += nom

        prob_vector = np.array(nom_list)/sum
        return prob_vector

    def acceptByChance(self,softmax_list):
        pointer = random.random()*sum(softmax_list)
        tier = 0
        idx = 0
        for value in softmax_list:
            if pointer >= tier and pointer < tier+value:
                return idx
            else:
                tier += value
                idx += 1
        return -1



    def getMean(self, reward_list):
        new_list = []
        for rewards in reward_list:
            new_list.append(sum(rewards)/len(rewards))

        return new_list


    def sent2tokens(self, sent_str):
        if self.without_stopwords and self.stem:
            return util.sent2stokens_wostop(sent_str, self.stemmer, self.stoplist, self.language)
        elif self.without_stopwords == False and self.stem:
            return util.sent2stokens(sent_str, self.stemmer, self.language)
        elif self.without_stopwords and self.stem == False:
            return util.sent2tokens_wostop(sent_str, self.stoplist, self.language)
        else:  # both false
            return util.sent2tokens(sent_str, self.language)


    # YG 14 Aug
    # do not merge all documents; instead, select top tf-idf tokens across documents
    def getTopTfidfTokens(self):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.stemmed_docs_list)
        tokens = vectorizer.get_feature_names()
        (xx, yy) = tfidf_matrix.nonzero()
        xx = xx.tolist()
        yy = yy.tolist()

        top_list = []
        while len(top_list) < self.top_ngrams_num:
            largest_value = -999
            idx_ii = -1
            for ii in range(len(xx)):
                cand = tfidf_matrix[xx[ii], yy[ii]]
                if yy[ii] not in top_list and cand > largest_value:
                    largest_value = cand
                    idx_ii = ii
            top_list.append(yy[idx_ii])
            del xx[idx_ii]
            del yy[idx_ii]

        for idx in top_list:
            self.top_ngrams_list.append(tokens[idx])

    def getTopGrams(self):
        sent_list = []
        for sent in self.sentences:
            sent_list.append(sent.untokenized_form)
        #for nn in sorted(sent_list):
            #print(nn)
        self.top_ngrams_list = util.getTopNgrams(sent_list, self.stemmer, self.language, self.stoplist, 2,
                                                 self.top_ngrams_num)

    def produceSummary(self,deep_model,model_idx):
        print('training finishes. start to produce a summary.')
        state = State(self.sum_token_length, self.base_length, len(self.sentences), self.state_block_num, self.language)

        # select sentences greedily
        while state.terminal_state == 0:
            new_sent_id = self.getGreedySent(state,deep_model)
            if new_sent_id == 0:
                break
            else:
                state.updateState(new_sent_id - 1, self.sentences)

        # if the selection terminates by 'finish' action
        if new_sent_id == 0:
            assert len(''.join(state.draft_summary_list).split(' ')) <= self.sum_token_length
            return state.draft_summary_list

        # else, the selection terminates because of over-length; thus the last selected action is deleted
        else:
            return state.draft_summary_list[:-1]

    def load_data(self):
        for doc_id, doc in enumerate(self.docs):
            doc_name, doc_sents = doc
            doc_tokens_list = []
            for sent_id, sent_text in enumerate(doc_sents):
                token_sent = word_tokenize(sent_text, self.language)
                current_sent = Sentence(token_sent, doc_id, sent_id + 1)
                untokenized_form = untokenize(token_sent)
                current_sent.untokenized_form = untokenized_form
                current_sent.length = len(untokenized_form.split(' '))
                self.sentences.append(current_sent)
                sent_tokens = self.sent2tokens(untokenized_form)
                doc_tokens_list.extend(sent_tokens)
                stemmed_form = ' '.join(sent_tokens)
                self.stemmed_sentences_list.append(stemmed_form)
            self.stemmed_docs_list.append(' '.join(doc_tokens_list))
        print('total sentence num: ' + str(len(self.sentences)))

        self.state_length_computer = StateLengthComputer(self.state_block_num,self.base_length,len(self.sentences))
        self.top_ngrams_num = self.state_length_computer.getStatesLength(self.state_block_num)
        self.vec_length = self.state_length_computer.getTotalLength()
        self.weights = np.zeros(self.vec_length)  # np.random.rand(self.vec_length)
        print('total vector length: {}'.format(self.vec_length))
        print('in total {} state blocks, each block length is as follows:'.format(self.state_block_num))
        for i in range(1,self.state_block_num+1):
            print(self.state_length_computer.getStatesLength(i))


    def getGreedySent(self, state, deep_model):
        vec_list = []
        str_vec_list = []
        current_state_vec = state.getSelfVector(self.top_ngrams_list,self.sentences,self.state_type)

        if len(state.available_sents) == 1:
            return 0

        for act_id in state.available_sents:
            # for action 'finish', the reward is the terminal reward
            if act_id == 0:
                vec_variable = Variable(torch.from_numpy(np.array(current_state_vec)).float())
                terminate_reward = deep_model(vec_variable.unsqueeze(0)).data.numpy()[0][0]
            # otherwise, the reward is 0, and value-function can be computed using the weight
            else:
                temp_state_vec = state.getNewStateVec(act_id-1, self.top_ngrams_list, self.sentences, self.state_type)#, current_state_vec)
                vec_list.append(temp_state_vec)
                str_vec = ''
                for ii,vv in enumerate(temp_state_vec):
                    if vv != 0.:
                        str_vec += '{}:{};'.format(ii,vv)
                str_vec_list.append(str_vec)

        # get action that results in highest values
        variable = Variable(torch.from_numpy(np.array(vec_list)).float())
        values = deep_model(variable)
        values_list = values.data.numpy()
        #print('vectors list: ')
        #for vv in str_vec_list:
            #print(vv)
        max_value = float('-inf')
        max_idx = -1
        for ii,value in enumerate(values_list):
            if value[0] > max_value:
                max_value = value[0]
                max_idx = ii

        if terminate_reward > max_value:
            return 0
        else:
            return state.available_sents[max_idx+1]


    def deepTrain(self, vec_list, reward, last_deep_model):
        value_variables = self.deep_model(Variable(torch.from_numpy(np.array(vec_list)).float()))
        old_value_variables = last_deep_model(Variable(torch.from_numpy(np.array(vec_list)).float()))
        value_list = old_value_variables.data.numpy()
        target_list = []
        for idx in range(len(value_list)-1):
            target_list.append(self.gamma*value_list[idx+1][0])
        target_list.append(reward)
        target_variables = Variable(torch.from_numpy(np.array(target_list)).float())

        loss_fn = torch.nn.MSELoss(size_average=False)
        loss = loss_fn(value_variables,target_variables)
        #print('loss: {}'.format(loss.data[0]))

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss.data[0]

if __name__ == '__main__':

    def printResult(summary, read_model_name=''):
        token_num = 0
        for sent in summary:
            token_num += len(sent.split(' '))

        print('summary in topic {}, sent num {}, char num {}: \n {}'.format(topic, len(summary), token_num, summary))
        cnt = 0
        for model in models:
            model_name = model[0][model[0].rfind('/') + 1:]
            if read_model_name != '' and model_name != read_model_name:
                continue
            print("Model: %s" % model_name)
            R1_1, R2_1, RL, R4_1 = rouge(' '.join(summary), [model], summary_len)
            R1_list.append(R1_1)
            R2_list.append(R2_1)
            RL_list.append(RL)
            R4_list.append(R4_1)
            print('rouge scores, rouge_1 : {}; rouge_2 : {}; rouge-L: {}; rouge_4: {}'.format(R1_1, R2_1, RL, R4_1))
            cnt += 1

        print('########')
        print('R1 mean: {}, std-dev: {}'.format(np.mean(R1_list), np.std(R1_list)))
        print('R2 mean: {}, std-dev: {}'.format(np.mean(R2_list), np.std(R2_list)))
        print('RL mean: {}, std-dev: {}'.format(np.mean(RL_list), np.std(RL_list)))
        print('R4 mean: {}, std-dev: {}'.format(np.mean(R4_list), np.std(R4_list)))
        print('########')


    datasets_processed_path = '/home/gao/PycharmProjects/sampling_summaries/processed_data'
    data_set = 'DUC2002'
    summary_len = 100
    base_length = 200
    scan_round = 5000
    block_num = 1
    sample_max_num = 1000
    alpha = 0.001
    opt_type = 'adam'
    temp = 5.
    temp_decay = 0.9
    interval = 500

    reader = CorpusReader(datasets_processed_path)
    data = reader.get_data(data_set, summary_len)

    rouge = Rouge('/home/gao/workspace/v43-12Sep/ROUGE/ROUGE-RELEASE-1.5.5/')

    # reader2 = CorpusReader(datasets_processed_path, "parse")
    # parse_data = reader2.get_data(data_set, 100)

    R1_list = []
    R2_list = []
    RL_list = []
    R4_list = []

    topic_cnt = 0
    for topic, docs, models in data:
        topic_cnt += 1
        if topic_cnt > 5:
            continue

        summarizer = DeepTDAgent('english', data_set, scan_round, topic, models,
                                 alpha, opt_type, temp, temp_decay,
                                 sample_max_num, interval, base_length, block_num)
        summary = summarizer(docs)

        model_names = summary[-1]
        for (i, summy) in enumerate(summary[:-1]):
            print('++++++Result for Summary {}, TOPIC {} ++++++'.format(i + 1, topic_cnt))
            printResult(summy, model_names[i])

    rouge.clean()
