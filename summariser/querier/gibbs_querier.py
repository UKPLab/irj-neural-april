import random
from summariser.querier.logistic_reward_learner import *
from summariser.utils.misc import normaliseList, softmaxSample

class GibbsQuerier:

    def __init__(self,summary_vectors,heuristic_values,learnt_weight=0.5):
        self.summary_vectors = summary_vectors
        self.reward_learner = LogisticRewardLearner()
        self.learnt_values = [0.]*len(summary_vectors)
        self.heuristics = heuristic_values
        self.learnt_weight = learnt_weight

    def inLog(self,sum1,sum2,log):
        for entry in log:
            if [sum1,sum2] in entry:
                return True
            elif [sum2,sum1] in entry:
                return True

        return False


    def getQuery(self,log):
        mix_values = self.getMixReward()

        sum_idx1 = softmaxSample(mix_values,1.0)
        sum_idx2 = softmaxSample(mix_values,-1.0)

        ### ensure the sampled pair has not been queried before
        while(self.inLog(sum_idx1,sum_idx2,log)):
            sum_idx1 = softmaxSample(mix_values,1.0)
            sum_idx2 = softmaxSample(mix_values,-1.0)

        if random.random() > 0.5:
            return sum_idx1, sum_idx2
        else:
            return sum_idx2, sum_idx1

    def updateRanker(self,pref_log):
        self.reward_learner.train(pref_log,self.summary_vectors)
        self.learnt_values = self.getReward()

    def getReward(self):
        values = [np.dot(self.reward_learner.weights,vv) for vv in self.summary_vectors]
        return normaliseList(values)

    def getMixReward(self,learnt_weight=-1):
        if learnt_weight == -1:
            learnt_weight = self.learnt_weight

        mix_values = np.array(self.learnt_values)*learnt_weight+np.array(self.heuristics)*(1-learnt_weight)

        return normaliseList(mix_values)




