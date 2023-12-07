import torch
import numpy as np
from dataset import Dataset
from scripts import shredFacts
from mgmf_distmult import MGMF_DistMult
from measure import Measure
import torch.nn.functional as F
from torch import nn
class LSTM_Regression(nn.Module):

    def __init__(self, input_size, hidden_size, output_size=1, num_layers=4):
        super().__init__()
        self.lstm = nn.GRU(input_size, hidden_size, num_layers)
        self.fc1 = nn.Linear(hidden_size, 36)
        self.fc2 = nn.Linear(50, 36)

    def forward(self, _x):
        x, _ = self.lstm(_x)
        x = self.fc1(x)
        return x


class LSTM_Regression1(nn.Module):

    def __init__(self, input_size, hidden_size, output_size=1, num_layers=4):
        super().__init__()
        self.lstm = nn.GRU(input_size, hidden_size, num_layers)
        self.fc1 = nn.Linear(hidden_size, 50)
        self.fc2 = nn.Linear(50, 36)

    def forward(self, _x):
        x, _ = self.lstm(_x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Tester:
    def __init__(self, dataset, model_path, valid_or_test):
        self.model = torch.load(model_path)
        self.model.eval()
        self.dataset = dataset
        self.valid_or_test = valid_or_test
        self.measure = Measure()
        
    def getRank(self, sim_scores):#assuming the test.txt fact is the first one
        return (sim_scores > sim_scores[0]).sum() + 1

    def getRank1(self, sim_scores):#assuming the test.txt fact is the first one

        temp=0
        max=-1
        for i in range(len(sim_scores)-1):
            if temp< sim_scores[i]:
                temp=sim_scores[i]
                max=i
        return max

    def getRank2(self, sim_scores):#assuming the test.txt fact is the first one

        temp=0
        ind=-5
        max=100000000000000
        for i in range(len(sim_scores)-1):
            temp=abs(sim_scores[i]-sim_scores[len(sim_scores)-1])
            if temp<max:
                max=temp
                ind=i
        return ind

    def replaceAndShred(self, fact, raw_or_fil, head_or_tail):
        head, rel, tail, years, months, days, mm, mmend = fact
        if head_or_tail == "head":

            ret_facts = [(i, rel, tail, years, months, days,mm, mmend) for i in range(self.dataset.numEnt())]


        if head_or_tail == "tail":
            ret_facts = [(head, rel, i, years, months, days,mm, mmend) for i in range(self.dataset.numEnt())]


        if raw_or_fil == "raw":
            ret_facts = [tuple(fact)] + ret_facts

        elif raw_or_fil == "fil":
            ret_facts = [tuple(fact)] + list(set(ret_facts) - self.dataset.all_facts_as_tuples)
        return shredFacts(np.array(ret_facts))

    def test(self):
        for i, fact in enumerate(self.dataset.data[self.valid_or_test]):
            settings = ["fil"]
            for raw_or_fil in settings:
                for head_or_tail in ["head", "tail"]:
                    heads, rels, tails, years, months, days, mm, mmend  = self.replaceAndShred(fact, raw_or_fil, head_or_tail)
                    sim_scores = self.model(heads, rels, tails, years, months, days, mm, mmend).cpu().data.numpy()
                    rank = self.getRank(sim_scores)
                    self.measure.update(rank, raw_or_fil)

        self.measure.print_()
        self.measure.normalize(len(self.dataset.data[self.valid_or_test]))
        self.measure.print_()

        return self.measure.mrr["fil"]