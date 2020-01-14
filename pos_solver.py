#!/usr/local/bin/python3
###################################
# CS B551 Fall 2019, Assignment #3
#
# Your names and user ids: Yash Jain: yashjain, Dhruv Yadwadkar: dyadwad, Vrinda Mathur: vrmath 
#
# (Based on skeleton code by D. Crandall)
#


from random import random 
import math

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def __init__(self):
        self.sentence=[]
        self.initial={}        
        self.count={}        
        self.transitional={}        
        self.emission={}
        self.all_words=[]
        self.all_tags=[]
        self.start=[]
        self.end=[]
        self.mc={}
        self.unique_words={}
        self.unique_tags={}
        self.initi={}
        self.save=[]
        
  
    def posterior(self, model, sentence, label):
        if model == "Simple" or model== "HMM" or model=="Complex":
            numerator=1 
                       
            for i, word in enumerate(sentence):
                l=label[i]
                if i==0:
                    numerator=math.log(self.initial[l])                        
                else:
                    numerator+=math.log(self.transitional[label[i-1]][l]) + math.log(self.emission[l][word])                        
                denominator=math.log(sum(v for _,v in self.count[word].items())/len(sentence)) 
                #print(numerator)
                #print(denominator)                  
            return numerator-denominator
        else:
            print("Unknown algo!")
    
    def initialisation(self,one, two, dict, dict2):
        for a, b in zip(one, two):
           if a not in dict:
               dict[a]={}
               for ta in dict2.keys():
                   #Assigning very small value to relation between 'a' and 'ta' that does exist in training dataset
                   #generalisation
                   dict[a][ta]=1e-12
           dict[a][b]+=1
        return dict
           
    # Do the training!
    def train(self, data):
        for sentence in data:
            for w in sentence[0]:
                if w not in self.unique_words:
                     self.unique_words[w]=0
                self.all_words.append(w)
            for i, t in enumerate(sentence[1]): 
                self.all_tags.append(t)
                if t not in self.unique_tags:
                    self.unique_tags[t]=0
                self.unique_tags[t]+=1
                if i==0:
                    self.start.append(t)
                elif i==len(sentence)-1:
                    self.end.append(t)
    
   
        
        self.mc=self.initialisation(self.start, self.end, self.mc, self.unique_tags)
        self.mc=self.probability(self.mc)
        
        self.count=self.initialisation(self.all_words, self.all_tags, self.count, self.unique_tags)
        self.count=self.probability(self.count)
        
        self.emission=self.initialisation(self.all_tags, self.all_words, self.emission, self.unique_words)                
        self.emission=self.probability(self.emission)                 
                                      
        for t,_ in self.unique_tags.items():
                self.initial[t]=0                
        for sentence in data:
                self.initial[sentence[1][0]]+=1/len(data)
          #Unnecessary
#         for k, v in self.initial.items():
#             c=v/sum(self.initial.values())
#             self.initial[k]=c

        for i, t in enumerate(self.all_tags):
            if t not in self.transitional:
                self.transitional[t]={}
                for ta,_ in self.unique_tags.items():
                    #Assigning a very small value to transitions between tags that may not exist in training
                    self.transitional[t][ta]=1e-12
            if i != len(self.all_tags)-1:
                self.transitional[t][self.all_tags[i+1]]+=1
        self.transitional=self.probability(self.transitional)
        
        #Previously written code for training the data, was obviously going to take too long 
#         for s, e in zip(start, end):
#             if s not in self.mc:
#                 self.mc[s]={}
#                 for ta in self.unique_tags:
#                     self.mc[s][ta]=0
#             self.mc[s][e]+=1            
#         for (w, t) in zip(self.all_words, self.all_tags):
#             if w not in self.count:
#                 self.count[w]={}
#                 for ta, _ in self.unique_tags.items():
#                     self.count[w][ta]=0                           
#             self.count[w][t]+=1
        
#             if t not in self.emission:
#                 self.emission[t]={}
#                 for word in self.all_words:
#                     self.emission[t][word]=1e-12
#             self.emission[t][w]+=1      
       
#        print("Count", self.count)
#        print("Initial", self.initial)
#        print("Transitional", self.transitional)
#        print("MC", self.mc)
#        print("Emission", self.emission)

    def probability(self, diction):
        for k, v in diction.items():
                l= [(ki, vi) for ki, vi in v.items()]
                s=sum(x[1] for x in l) 
                for ki, vi in v.items():
                    v[ki]=vi/s
        return diction         
    
    #POS tagged based on the frequency of the word being paired with a certain speech tag
    #arg max_si_P(Tag_i=tag_i|W)
    def simplified(self, sentence):
        l=[]
        tag=[]
        for word in sentence:
            l=[]
            if word not in self.count:
                self.count[word]={}
                for ta in self.unique_tags:
                    self.count[word][ta]=1e-12          
            for k, v in self.count[word].items():
                l.append((k,v))
            tag.append(max(l, key=lambda x: x[1])[0])
        #print("Sentence", sentence)        
        #print("Simple", tag)
        self.save=tag
        return tag
    
    #POS tagged based on Viterbi algorithm
    #(s∗1, . . . , s∗N) = arg max_s1,...,sN_P(Si=si|W) 
    def hmm_viterbi(self, sentence):
        tag=[]
        viterbi=[]       
        for i, word in enumerate(sentence):            
            f=[]
            if i==0:
                if word not in self.all_words:
                    for ta,n in self.unique_tags.items():
                        f.append((ta, n*1e-12)) 
                else:           
                    for k, v in self.initial.items():                    
                        f.append((k, v*self.emission[k][word]))
                v=max(f, key=lambda x: x[1])
                viterbi.append(v[1])
                tag.append(v[0])
               
            else:                
                if word not in self.all_words:
                    for ta in self.unique_tags:
                        self.emission[ta][word]=1e-12
                k=tag[i-1]
                for ki, vi in self.transitional[k].items():
                    f.append((ki, vi*self.emission[ki][word]*viterbi[i-1]))
                v=max(f, key=lambda x: x[1])                   
                viterbi.append(v[1])
                tag.append(v[0]) 
        #print("HMM", tag)   
        return tag
     
    def complex_mcmc(self, sentence):
        #initial_states=self.save
        #using the state of tags returned by Naive Bayes method does not give great accuracy, gives lesser accuracy of sentences                 
        initial_states=['noun']*len(sentence)        
        updated=initial_states
        
        gen_sample=[[initial_states]]
        #generating sample of 1000 samples
        for i in range(1000):
            initial_states=updated
            for j, word in enumerate(sentence):
                pos_prob=[]          
                for m, tag in enumerate(self.unique_tags.keys()):
                    if word not in self.unique_words:
                        self.emission[tag][word]=1e-12
                    p=self.emission[tag][word]
                    #Markov Chain: Dependent only on previous and next tag
                    #print(initial_states)
                    
                    #the 388th sentence is only one word
                    if len(sentence)==1:
                        p*= self.initial[tag]                                      
                    elif j == 0:
                        t=tag
                        #P(tag_i) included for the first word of the sentence
                        #P(tag_i) * P(W_i|tag_i) * P(tag_i+1|tag_i)
                        #Mistake: P(tag_i) * P(W_i|tag_i) * P(tag_i|tag_i+1)
                        p*= self.initial[tag]*self.transitional[tag][initial_states[j+1]]
                    elif j==len(sentence)-1:
                        #P(tag_i|tag_i-1) * P(W_i|tag_i) * P(tag_i|tag_0)
                        p*= self.transitional[initial_states[j-1]][tag]*self.mc[t][tag]                    
                    else:
                        #P(tag_i+1|tag_i) * P(tag_i|tag_i-1) * P(W_i|tag_i)
                        #Mistake: P(tag_i|tag_i+1) * P(tag_i|tag_i-1) * P(W_i|tag_i)
                        p*= self.transitional[tag][initial_states[j+1]]*self.transitional[initial_states[j-1]][tag]
       
                    pos_prob.append((tag,p))
                                                         
                s=sum(x[1] for x in pos_prob)                                               
                pos_prob=[(y[0],y[1]/s) for y in pos_prob]  
                #Mistake: Assigned non-random tag based on the max probability of certain tag of resultant
                # of the longer range dependencies of the Bayes Net                          
                #initial_states[j]=max(pos_prob,key=lambda x: x[1])[0]
                
                #Monte Carlo: "Taking a random walk on" the sentence (assigning random tags) 
                #Random float between 0 and 1                 
                randomprob=random()                
                cs=0
                temp=[]
                tag_random='' 
                    
                for l in pos_prob:
                    cs+=l[1]
                    l=(l[0], cs)
                    #Saves the cumulative frequencies of a POS tags post normalization 
                    temp.append(l)                 
 
                tag_random=''
                for e,l in enumerate(temp):
                    if randomprob <= l[1] and randomprob >= temp[e-1][1]:
                       tag_random=l[0]
                #if value remains unchanged, implies it lesser than the first cumulative value 
                if tag_random=='':                
                     tag_random=pos_prob[0][0]
                 
                #assigned random tag on jth position of sentence
                initial_states[j]=tag_random
            
            updated=initial_states
            gen_sample.append(updated)
        #final sample to calculate max marginal probabilities is the last 800 samples of the generated 1000 samples     
        return self.calc_mm(gen_sample[200: ], sentence)
                 
    #Calculating the max marginal for each word, 
    #s_i= arg max_si_P(Si=si|W)               
    def calc_mm(self, sample, sentence):
        mm_dict={}        
        tg=[]       
        
        for i, s in enumerate(sample):
            #print(i,s)
            for t, word in zip(s, sentence):
                if word not in mm_dict:
                    mm_dict[word]={}
                for tag in self.unique_tags:
                    mm_dict[word][tag]=0
                mm_dict[word][t]+=1
        #mm_dict=self.probability(mm_dict)
        #print(mm_dict)
        for word in sentence:
            a=[]
            for ta, n in mm_dict[word].items():
                a.append((ta, n))
            tg.append(max(a, key=lambda x:x[1])[0])
        return tg      
                
     
    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")
