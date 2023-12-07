import gc

import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from params import Params
from dataset import Dataset

class MGMF_DistMult(torch.nn.Module):


    def __init__(self, dataset, params):
        super(MGMF_DistMult, self).__init__()
        self.dataset = dataset
        self.params = params

        self.ent_embs      = nn.Embedding(dataset.numEnt(), 36).cuda()
        self.rel_embs      = nn.Embedding(dataset.numRel(), 36).cuda()
        self.create_time_embedds()
        self.time_nl = torch.sin
        self.time_nl1 = torch.cos

        nn.init.xavier_uniform_(self.ent_embs.weight)
        nn.init.xavier_uniform_(self.rel_embs.weight)
    def get_init_embeddings(self,entinit):
        lstent = []
        lstrel = []
        with open(entinit) as f:
            for line in f:
                tmp = [float(val) for val in line.strip().split()]
                lstent.append(tmp)
        print(lstent)
        return np.array(lstent, dtype=np.float32)

    def create_time_embedds(self):
        self.time_nl = torch.sin
        self.m_freq = nn.Embedding(self.dataset.numRel(), 64).cuda() #月
        self.d_freq = nn.Embedding(self.dataset.numRel(), 64).cuda() #日
        self.y_freq = nn.Embedding(self.dataset.numRel(), 64).cuda() # 年
        self.mm_freq = nn.Embedding(self.dataset.numRel(), 64).cuda()  # 分钟
        self.mmend_freq = nn.Embedding(self.dataset.numRel(), 64).cuda()  # 分钟

        self.mm_freq22 = nn.Embedding(25, 64).cuda()  # 分钟
        self.mmend_freq22= nn.Embedding(61, 64).cuda()  # 分钟
        nn.init.xavier_uniform_(self.m_freq.weight)
        nn.init.xavier_uniform_(self.d_freq.weight)
        nn.init.xavier_uniform_(self.y_freq.weight)
        nn.init.xavier_uniform_(self.mm_freq.weight)
        nn.init.xavier_uniform_(self.mmend_freq.weight)

        nn.init.xavier_uniform_(self.mm_freq22.weight)
        nn.init.xavier_uniform_(self.mmend_freq22.weight)

        self.m_phi = nn.Embedding(self.dataset.numRel(), 64).cuda()
        self.d_phi = nn.Embedding(self.dataset.numRel(), 64).cuda()
        self.y_phi = nn.Embedding(self.dataset.numRel(), 64).cuda()
        self.mm_phi = nn.Embedding(self.dataset.numRel(), 64).cuda()
        self.mmend_phi = nn.Embedding(self.dataset.numRel(), 64).cuda()

        self.mm_phi22 = nn.Embedding(25, 64).cuda()
        self.mmend_phi22 = nn.Embedding(61, 64).cuda()
        nn.init.xavier_uniform_(self.m_phi.weight)
        nn.init.xavier_uniform_(self.d_phi.weight)
        nn.init.xavier_uniform_(self.y_phi.weight)
        nn.init.xavier_uniform_(self.mm_phi.weight)
        nn.init.xavier_uniform_(self.mmend_phi.weight)

        nn.init.xavier_uniform_(self.mm_phi22.weight)
        nn.init.xavier_uniform_(self.mmend_phi22.weight)

        self.m_amp = nn.Embedding(self.dataset.numRel(), 64).cuda()
        self.y_amp = nn.Embedding(self.dataset.numRel(), 64).cuda()
        self.d_amp = nn.Embedding(self.dataset.numRel(), 64).cuda()
        self.mm_amp = nn.Embedding(self.dataset.numRel(),64).cuda()
        self.mmend_amp = nn.Embedding(self.dataset.numRel(), 64).cuda()

        nn.init.xavier_uniform_(self.m_amp.weight)
        nn.init.xavier_uniform_(self.d_amp.weight)
        nn.init.xavier_uniform_(self.y_amp.weight)
        nn.init.xavier_uniform_(self.mm_amp.weight)
        nn.init.xavier_uniform_(self.mmend_amp.weight)

        self.m_freq1 = nn.Embedding(self.dataset.numRel(), 64).cuda()  # 月
        self.d_freq1 = nn.Embedding(self.dataset.numRel(), 64).cuda()  # 日
        self.y_freq1 = nn.Embedding(self.dataset.numRel(), 64).cuda()  # 年
        self.mm_freq1 = nn.Embedding(self.dataset.numRel(), 64).cuda()  # 分钟
        self.mmend_freq1 = nn.Embedding(self.dataset.numRel(), 64).cuda()  # 分钟

        nn.init.xavier_uniform_(self.m_freq1.weight)
        nn.init.xavier_uniform_(self.d_freq1.weight)
        nn.init.xavier_uniform_(self.y_freq1.weight)
        nn.init.xavier_uniform_(self.mm_freq1.weight)
        nn.init.xavier_uniform_(self.mmend_freq1.weight)

        self.m_phi1 = nn.Embedding(self.dataset.numRel(), 64).cuda()
        self.d_phi1 = nn.Embedding(self.dataset.numRel(), 64).cuda()
        self.y_phi1= nn.Embedding(self.dataset.numRel(), 64).cuda()
        self.mm_phi1 = nn.Embedding(self.dataset.numRel(), 64).cuda()
        self.mmend_phi1 = nn.Embedding(self.dataset.numRel(), 64).cuda()

        nn.init.xavier_uniform_(self.m_phi1.weight)
        nn.init.xavier_uniform_(self.d_phi1.weight)
        nn.init.xavier_uniform_(self.y_phi1.weight)
        nn.init.xavier_uniform_(self.mm_phi1.weight)
        nn.init.xavier_uniform_(self.mmend_phi1.weight)

        self.m_amp1 = nn.Embedding(self.dataset.numRel(), 64).cuda()
        self.y_amp1 = nn.Embedding(self.dataset.numRel(), 64).cuda()
        self.d_amp1 = nn.Embedding(self.dataset.numRel(), 64).cuda()
        self.mm_amp1 = nn.Embedding(self.dataset.numRel(), 64).cuda()
        self.mmend_amp1 = nn.Embedding(self.dataset.numRel(), 64).cuda()

        nn.init.xavier_uniform_(self.m_amp1.weight)
        nn.init.xavier_uniform_(self.d_amp1.weight)
        nn.init.xavier_uniform_(self.y_amp1.weight)
        nn.init.xavier_uniform_(self.mm_amp1.weight)
        nn.init.xavier_uniform_(self.mmend_amp1.weight)

    def get_time_embedd(self, entities, year, month, day,mms,mmsend):

        entities = entities.cpu().numpy()
        year = year.cpu().numpy()
        month = month.cpu().numpy()
        day = day.cpu().numpy()
        mms = mms.cpu().numpy()
        mmsend = mmsend.cpu().numpy()

        index = []
        mmsx1=[]

        for i in range(len(mms)):
            if mms[i] > 100:
                index.append(i)
                mmsx1.append(mms[i]-100)

        entities1=torch.from_numpy(np.array(entities)[index]).cuda()
        temp1=np.delete(entities, index, axis=0)
        entities = torch.from_numpy(temp1).cuda()
        entities01 = torch.from_numpy(temp1.reshape(-1)).long().cuda()

        year1 = torch.from_numpy(np.array(year)[index]).cuda()
        year = torch.from_numpy(np.delete(year, index, axis=0)).cuda()

        month1 = torch.from_numpy(np.array(month)[index]).cuda()
        month = torch.from_numpy(np.delete(month, index, axis=0)).cuda()

        day1 = torch.from_numpy(np.array(day)[index]).cuda()
        day11 = torch.from_numpy(np.array(day)[index].reshape(-1)).long().cuda()

        tempday= np.delete(day, index, axis=0)
        day = torch.from_numpy(tempday).cuda()
        day0 = torch.from_numpy(tempday.reshape(-1)).long().cuda()


        mms1 = torch.from_numpy(np.array(mmsx1)).cuda()
        tempmms1=np.array(mmsx1)
        mms11 = torch.from_numpy(tempmms1.reshape(-1)).long().cuda()

        tempmms=np.delete(mms, index, axis=0)
        mms = torch.from_numpy(tempmms).cuda()
        mms0 = torch.from_numpy(tempmms.reshape(-1)).long().cuda()



        mmsend1 = torch.from_numpy(np.array(mmsend)[index]-100).cuda()
        tempmmsend1=np.array(mmsend)[index]-100
        mmsend11 = torch.from_numpy(tempmmsend1.reshape(-1)).long().cuda()



        temphour1=mmsend11.cpu().numpy()
        temphour=mmsend11-mms11
        temphour=temphour.cpu().numpy()


        tempmmsend=np.delete(mmsend, index, axis=0)
        mmsend = torch.from_numpy(tempmmsend).cuda()
        mmsend0 = torch.from_numpy(tempmmsend.reshape(-1)).long().cuda()





        datayear=np.array([0])
        datayear1 = np.array([0])
        datamonth = np.array([0])

        datahour = np.array([mms])
        dataminute = np.array([mmsend])


        y = self.y_amp(entities) * self.time_nl(self.y_freq(entities) * year + self.y_phi(entities)).cuda()
        # y = self.y_amp(entities) * self.time_nl(self.y_freq(entities) * year + self.y_phi(entities)).cuda()

        m = self.m_amp(entities) * self.time_nl(self.y_freq(entities) * month + self.m_phi(entities)).cuda()
        d = self.d_amp(entities) * self.time_nl(self.d_freq(entities) * day + self.d_phi(entities)).cuda()


        mm = self.mm_amp(entities) * self.time_nl(self.mm_freq(entities) * mms + self.mm_phi(entities)).cuda()
        mmend = self.mmend_amp(entities) * self.time_nl(self.mmend_freq(entities) * mmsend + self.mmend_phi(entities)).cuda()

        y1 = self.y_amp1(entities) * self.time_nl1(self.y_freq1(entities) * year + self.y_phi1(entities)).cuda()
        m1 = self.m_amp1(entities) * self.time_nl1(self.m_freq1(entities) * month + self.m_phi1(entities)).cuda()
        d1 = self.d_amp1(entities) * self.time_nl1(self.d_freq1(entities) * day + self.d_phi1(entities)).cuda()
        mm1 = self.mm_amp1(entities) * self.time_nl1(self.mm_freq1(entities) * mms + self.mm_phi1(entities)).cuda()
        mmend1 = self.mmend_amp1(entities) * self.time_nl1(self.mmend_freq1(entities) * mmsend + self.mmend_phi1(entities)).cuda()

        dataday = 0
        mmend2 = torch.zeros([1, 64]).cuda()
        if entities1.shape[0] >0:
            y2 = self.y_amp(entities1) * self.time_nl(self.y_freq(entities1) * year1 + self.y_phi(entities1)).cuda()
            m2 = self.m_amp(entities1) * self.time_nl(self.m_freq(entities1) * month1 + self.m_phi(entities1)).cuda()
            d2 = self.d_amp(entities1) * self.time_nl(self.d_freq(entities1) * day1 + self.d_phi(entities1)).cuda()

            y22 = self.y_amp1(entities1) * self.time_nl1(self.y_freq1(entities1) * year1 + self.y_phi1(entities1)).cuda()
            m22 = self.m_amp1(entities1) * self.time_nl1(self.m_freq1(entities1) * month1 + self.m_phi1(entities1)).cuda()
            d22 = self.d_amp1(entities1) * self.time_nl1(self.d_freq1(entities1) * day1 + self.d_phi1(entities1)).cuda()

            tensor6 = torch.full([entities1.shape[0],1], 6).cuda()
            tensor7 = torch.full([entities1.shape[0],1], 7).cuda()
            tensor8 = torch.full([entities1.shape[0],1], 8).cuda()
            tensor9 = torch.full([entities1.shape[0],1], 9).cuda()
            tensor10 = torch.full([entities1.shape[0],1], 10).cuda()
            tensor11= torch.full([entities1.shape[0],1], 11).cuda()
            tensor12 = torch.full([entities1.shape[0],1], 12).cuda()
            tensor13 = torch.full([entities1.shape[0],1], 13).cuda()
            tensor14 = torch.full([entities1.shape[0],1], 14).cuda()
            tensor15 = torch.full([entities1.shape[0],1], 15).cuda()
            tensor16 = torch.full([entities1.shape[0],1], 16).cuda()
            tensor17 = torch.full([entities1.shape[0],1], 17).cuda()
            tensor18 = torch.full([entities1.shape[0],1], 18).cuda()
            tensor19 = torch.full([entities1.shape[0],1], 19).cuda()
            tensor20 = torch.full([entities1.shape[0],1], 20).cuda()
            tensor21 = torch.full([entities1.shape[0],1], 21).cuda()
            tensor22 = torch.full([entities1.shape[0],1], 22).cuda()
            tensor23 = torch.full([entities1.shape[0],1], 23).cuda()
            tensor24 = torch.full([entities1.shape[0],1], 24).cuda()


            mm2 = self.mm_amp(entities1) * self.time_nl(self.mm_freq(entities1) * tensor6+ self.mm_phi((entities1)))\
                  +self.mm_amp(entities1) * self.time_nl(self.mm_freq(entities1) * tensor7+ self.mm_phi((entities1)))\
                  +self.mm_amp(entities1) * self.time_nl(self.mm_freq(entities1) * tensor8+ self.mm_phi((entities1)))\
                  +self.mm_amp(entities1) * self.time_nl(self.mm_freq(entities1) * tensor9+ self.mm_phi((entities1)))\
                  +self.mm_amp(entities1) * self.time_nl(self.mm_freq(entities1) * tensor10+ self.mm_phi((entities1)))\
                  +self.mm_amp(entities1) * self.time_nl(self.mm_freq(entities1) * tensor11+ self.mm_phi((entities1)))\
                  +self.mm_amp(entities1) * self.time_nl(self.mm_freq(entities1) * tensor12+ self.mm_phi((entities1)))\
                  +self.mm_amp(entities1) * self.time_nl(self.mm_freq(entities1) * tensor13+ self.mm_phi((entities1)))\
                  +self.mm_amp(entities1) * self.time_nl(self.mm_freq(entities1) * tensor14+ self.mm_phi((entities1)))\
                  +self.mm_amp(entities1) * self.time_nl(self.mm_freq(entities1) * tensor15+ self.mm_phi((entities1)))\
                  +self.mm_amp(entities1) * self.time_nl(self.mm_freq(entities1) * tensor16+ self.mm_phi((entities1)))\
                  +self.mm_amp(entities1) * self.time_nl(self.mm_freq(entities1) * tensor17+ self.mm_phi((entities1)))\
                  +self.mm_amp(entities1) * self.time_nl(self.mm_freq(entities1) * tensor18+ self.mm_phi((entities1)))\
                  +self.mm_amp(entities1) * self.time_nl(self.mm_freq(entities1) * tensor19+ self.mm_phi((entities1)))\
                  +self.mm_amp(entities1) * self.time_nl(self.mm_freq(entities1) * tensor20+ self.mm_phi((entities1)))\
                  +self.mm_amp(entities1) * self.time_nl(self.mm_freq(entities1) * tensor21+ self.mm_phi((entities1)))\
                  +self.mm_amp(entities1) * self.time_nl(self.mm_freq(entities1) * tensor22+ self.mm_phi((entities1)))\
                  +self.mm_amp(entities1) * self.time_nl(self.mm_freq(entities1) * tensor23+ self.mm_phi((entities1)))\
                  +self.mm_amp(entities1) * self.time_nl(self.mm_freq(entities1) * tensor24+ self.mm_phi((entities1)))

            mm22 = self.mm_amp1(entities1) * self.time_nl1(self.mm_freq1(entities1) * tensor6 + self.mm_phi1((entities1))) \
                  + self.mm_amp1(entities1) * self.time_nl1(self.mm_freq1(entities1) * tensor7 + self.mm_phi1((entities1))) \
                  + self.mm_amp1(entities1) * self.time_nl1(self.mm_freq1(entities1) * tensor8 + self.mm_phi1((entities1))) \
                  + self.mm_amp1(entities1) * self.time_nl1(self.mm_freq1(entities1) * tensor9 + self.mm_phi1((entities1))) \
                  + self.mm_amp1(entities1) * self.time_nl1(self.mm_freq1(entities1) * tensor10 + self.mm_phi1((entities1))) \
                  + self.mm_amp1(entities1) * self.time_nl1(self.mm_freq1(entities1) * tensor11 + self.mm_phi1((entities1))) \
                  + self.mm_amp1(entities1) * self.time_nl1(self.mm_freq1(entities1) * tensor12 + self.mm_phi1((entities1))) \
                  + self.mm_amp1(entities1) * self.time_nl1(self.mm_freq1(entities1) * tensor13 + self.mm_phi1((entities1))) \
                  + self.mm_amp1(entities1) * self.time_nl1(self.mm_freq1(entities1) * tensor14 + self.mm_phi1((entities1))) \
                  + self.mm_amp1(entities1) * self.time_nl1(self.mm_freq1(entities1) * tensor15 + self.mm_phi1((entities1))) \
                  + self.mm_amp1(entities1) * self.time_nl1(self.mm_freq1(entities1) * tensor16 + self.mm_phi1((entities1))) \
                  + self.mm_amp1(entities1) * self.time_nl1(self.mm_freq1(entities1) * tensor17 + self.mm_phi1((entities1))) \
                  + self.mm_amp1(entities1) * self.time_nl1(self.mm_freq1(entities1) * tensor18 + self.mm_phi1((entities1))) \
                  + self.mm_amp1(entities1) * self.time_nl1(self.mm_freq1(entities1) * tensor19 + self.mm_phi1((entities1))) \
                  + self.mm_amp1(entities1) * self.time_nl1(self.mm_freq1(entities1) * tensor20 + self.mm_phi1((entities1))) \
                  + self.mm_amp1(entities1) * self.time_nl1(self.mm_freq1(entities1) * tensor21 + self.mm_phi1((entities1))) \
                  + self.mm_amp1(entities1) * self.time_nl1(self.mm_freq1(entities1) * tensor22 + self.mm_phi1((entities1))) \
                  + self.mm_amp1(entities1) * self.time_nl1(self.mm_freq1(entities1) * tensor23 + self.mm_phi1((entities1))) \
                  + self.mm_amp1(entities1) * self.time_nl1(self.mm_freq1(entities1) * tensor24 + self.mm_phi1((entities1)))

            mmend2 = np.repeat(np.copy(mmend2.cpu().detach().numpy()), len(temphour), axis=0)
            mmend2 = torch.from_numpy(mmend2).cuda()
            mmend22=mmend2


            y=torch.cat((y, y2), 0)
            y1 = torch.cat((y1, y22), 0)

            m = torch.cat((m, m2), 0)
            m1 = torch.cat((m1, m22),0)

            d= torch.cat((d, d2), 0)
            d1 = torch.cat((d1, d22), 0)

            mm = torch.cat((mm, mm2), 0)
            mm1 = torch.cat((mm1, mm22), 0)

            mmend = torch.cat((mmend, mmend2), 0)
            mmend1 = torch.cat((mmend1, mmend22), 0)



        return (y+y1)/2+(m+m1)/2+(d+d1)/2+(mm+mm1)/2+(mmend+mmend1)/2, torch.cat((entities, entities1), 0)

    def getEmbeddings(self, heads, rels, tails, years, months, days, mm,mmend, intervals = None):

        years = years.view(-1,1)
        months = months.view(-1,1)
        days = days.view(-1,1)
        mm = mm.view(-1, 1)
        mmend = mmend.view(-1, 1)


        h_t, heads1 = self.get_time_embedd(heads, years, months, days,mm, mmend)

        h_r, rels1= self.get_time_embedd(rels, years, months, days, mm, mmend)

        t_t, tails1 = self.get_time_embedd(tails, years, months, days,mm, mmend)

        h, r, t = self.ent_embs(heads1), self.rel_embs(rels1), self.ent_embs(tails1)

        h = torch.cat((h,h_t), 1)

        r = torch.cat((r, h_r), 1)

        t = torch.cat((t,t_t), 1)

        return h,r,t

    def forward(self, heads, rels, tails, years, months, days, mm, mmend):
        a=[]
        h_embs, r_embs, t_embs = self.getEmbeddings(heads, rels, tails, years, months, days,mm,mmend)

        scores = (h_embs * r_embs) * t_embs
        scores = F.dropout(scores, p=self.params.dropout, training=self.training)
        scores = torch.sum(scores, dim=1)
        return scores
