
# coding: utf-8

# In[15]:
# NAme : Karan Malhotra
# Sr No: 14532
# M.tech Systems Engg

import numpy as np
import nltk
from nltk.corpus import gutenberg
from nltk.corpus import brown
from collections import Counter
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[16]:


def LM_model_kneser(train,test,k,d): # Bigram model with Kneser Ney smoothing
    
        train,held= train_test_split(train,test_size=0.1,random_state=4) 
    # held out data was used for tuning discount, after getting optimal discount value the tuning code section has been removed
	
	#Word tokenization for training sentences:
        train_words=[]
        for i in range(0,len(train)):
                for j in range(0,len(train[i])):
                        train_words.append(train[i][j])
    
    
    #making unigram_dictionary with the counts
        unigram_count=Counter(train_words)
    
    # Assigning some of words with unigram count as one to be UNK(Unkown words)
        for i in range(0,len(train_words)):
                if unigram_count[train_words[i]]==1:
                        train_words[i]='UNK'
                        k-=1
                if k==0:
                        break
        
    
    #updated unigrams with UNK included
        unigram_count=Counter(train_words)
    
    #construction of bigrams
        bigrams=[]
        bigram=nltk.bigrams(train_words)
        for bg in bigram:
                bigrams.append(bg)
        bigram_count=Counter(bigrams)
        bigram_count.pop(('</s>','<s>'))
    
    
    # finding P continuation and lambda(wi-1) for Kneser Ney Smoothing
        cont_words=[]
        prev_words=[]

        for key in bigram_count:
                cont_words.append(key[1])
                prev_words.append(key[0])
        cont_words_count=Counter(cont_words) 
        prev_words_count=Counter(prev_words)

   
        


    
    
    # Processing of Test Data
    
    #Word tokenization
        lambda_={}

        for key in unigram_count:
                if key!='</s>':
                        lambda_[key]=  (d*prev_words_count[key])/ (unigram_count[key])


        p_cont={}
        for key in unigram_count:
                if key!='<s>':
                        p_cont[key] =  cont_words_count[key] / (len(bigram_count))

        test_words=[]
        for i in range(0,len(test)):
                for j in range(0,len(test[i])):
                        if test[i][j] not in unigram_count:
                                test_words.append('UNK')
                        else:
                                test_words.append(test[i][j])
        N= len(test_words)-len(test)        
    #calculating perplexity of test data:
        perplexity=1
        for i in range(1,len(test_words)):
                if test_words[i]== "<s>"  and test_words[i-1]== '</s>':
                        prob=1
                else:
                        first=  max(  (bigram_count[(test_words[i-1],test_words[i])]-d) / unigram_count[test_words[i-1]]  ,   0 )
                        second=lambda_[test_words[i-1]] * p_cont[test_words[i]]
                        prob= first+second
                perplexity*= 1/(prob**(1/N))

        print('Perplexity with kneser ney smoothing is coming to be:',perplexity)    


# In[17]:


def LM_model_katz(train,test,k,gamma): #bigram model with katz backoff
        train,held= train_test_split(train,test_size=0.1,random_state=4)
    # held out data was used for tuning discount, after getting optimal discount value the tuning code section has been removed
	
	#Word tokenization for training sentences:
    
        train_words=[]
        for i in range(0,len(train)):
                for j in range(0,len(train[i])):
                        train_words.append(train[i][j])


        #making unigram_dictionary with the counts
        unigram_count=Counter(train_words)

    # Assigning some of words with unigram count as one to be UNK(Unkown words)

    

        for i in range(0,len(train_words)):
                if unigram_count[train_words[i]]==1:
                        train_words[i]='UNK'
                        k-=1
                if k==0:
                        break

        #updated unigrams with UNK included
        unigram_count=Counter(train_words)

        #construction of bigrams
        bigrams=[]
        bigram=nltk.bigrams(train_words)
        for bg in bigram:
                bigrams.append(bg)
        bigram_count=Counter(bigrams)
        bigram_count.pop(('</s>','<s>'))
    
    # for gamma tuning
    
    
    # for test set
        alpha=Counter(unigram_count.keys())
    
        for key in alpha:
                alpha[key]=0

        for key in bigram_count:
                alpha[key[0]]+= (bigram_count[key]-gamma)

        for w in alpha:
                alpha[w]= 1- ( alpha[w] / unigram_count[w] )
        summation=Counter(unigram_count.keys())
        for key in summation:
                summation[key]= sum( list(unigram_count.values() ) )

        for key in bigram_count:
                summation[key[0]]-=unigram_count[key[1]]
    #Word tokenization 
        test_words=[]
        for i in range(0,len(test)):
                for j in range(0,len(test[i])):
                        if test[i][j] not in unigram_count:
                                test_words.append('UNK')
                        else:
                                test_words.append(test[i][j])
        N= len(test_words) - len(test)

        perplexity=1
        for i in range(1,len(test_words)):
                if test_words[i]== "<s>"  and test_words[i-1]== '</s>':
                        prob=1
                else:
                        if bigram_count[(test_words[i-1],test_words[i])]>0:
                                prob =   (bigram_count[(test_words[i-1],test_words[i])] - gamma) / unigram_count[test_words[i-1]]
                        else:
                                prob =  ( alpha[test_words[i-1]] * unigram_count[test_words[i]] ) / summation[test_words[i-1]]
                perplexity*= 1/(prob**(1/N))

        print('Perplexity with KAtz Backoff is coming:',perplexity)


# In[18]:


def LM_model_trigram(train,test,k): #trigram model with stupid backoff
    #Word tokenization for training sentences:
    
        train_words=[]
        for i in range(0,len(train)):
                for j in range(0,len(train[i])):
                        train_words.append(train[i][j])


        #making unigram_dictionary with the counts
        unigram_count=Counter(train_words)

    # Assigning some of words with unigram count as one to be UNK(Unkown words)

    

        for i in range(0,len(train_words)):
                if unigram_count[train_words[i]]==1:
                        train_words[i]='UNK'
                        k-=1
                if k==0:
                        break

        #updated unigrams with UNK included
        unigram_count=Counter(train_words)

        #construction of bigrams
        bigrams=[]
        bigram=nltk.bigrams(train_words)
        for bg in bigram:
                bigrams.append(bg)
        bigram_count=Counter(bigrams)
        bigram_count.pop(('</s>','<s>'))
    #construction of trigrams
        trigrams=[]
        trigram=nltk.trigrams(train_words)
        for tg in trigram:
                trigrams.append(tg)
        trigram_count=Counter(trigrams)
    
    # stupid backoff on trigrams

    #Word tokenization 
        test_words=[]
        for i in range(0,len(test)):
                for j in range(0,len(test[i])):
                        if test[i][j] not in unigram_count:
                                test_words.append('UNK')
                        else:
                                test_words.append(test[i][j])
        N= len(test_words) - len(test)

        perplexity=1

        for i in range(2,len(test_words)):
                key=(test_words[i-2],test_words[i-1],test_words[i])
                if  trigram_count[key]>0 and bigram_count[(key[0],key[1])]>0:
                        prob= trigram_count[key]/ bigram_count[(key[0],key[1])]
                elif bigram_count[(key[1],key[2])]>0:
                        if key[1]=='</s>' and key[2]=='<s>':
                                prob=1
                        else:    
                                prob=  0.7*bigram_count[(key[1],key[2])] / unigram_count[key[1]]
                else:
                        prob= 0.7*unigram_count[key[2]] / sum(list(unigram_count.values()))
                perplexity*= (1/prob)**(1/N)
        print('Perplexity with Trigram Stupid backoff is coming as',perplexity)


# In[19]:


def test_train_selection(choice): # For Dataset combination selection S1,S2,S3,S4
    
        total_sent1=list(brown.sents())
        total_sent2=list(gutenberg.sents())
    # Start and end of sentence tagging
        for sent in total_sent1:
                sent.insert(0,"<s>")
                sent.insert(len(sent),"</s>")
    
        for sent2 in total_sent2:
                sent2.insert(0,"<s>")
                sent2.insert(len(sent2),"</s>")
        
        train1,test1= train_test_split(total_sent1,test_size=0.1,random_state=4)
        train2,test2= train_test_split(total_sent2,test_size=0.1,random_state=4)
    #optimal discount values are being passed which had been calculated through held out data    
        if choice==1:      # D1-train , D1- test
                LM_model_kneser(train1,test1,5500,0.8)
                LM_model_katz(train1,test1,5500,0.75)
                LM_model_trigram(train1,test1,5500)
        
        elif choice==2:    # D2-train , D2- test
                LM_model_kneser(train2,test2,5500,0.8)
                LM_model_katz(train2,test2,5500,0.75)
                LM_model_trigram(train2,test2,5500)
        elif choice==3:     # D1 + D2 train , D1- test
                LM_model_kneser(train1+total_sent2,test1,5500,0.75)
                LM_model_katz(train1+total_sent2,test1,5500,0.6)
                LM_model_trigram(train1+total_sent2,test1,5500)
        else:              # D1 + D2 train , D2- test
                LM_model_kneser(total_sent1+train2,test2,5500,0.8)
                LM_model_katz(total_sent1+train2,test2,5500,0.6)
                LM_model_trigram(total_sent1+train2,test2,5500)


# In[24]:


def random_sent(train):
    
        for sent in train:
                sent.insert(0,"<s>")
                sent.insert(0,"<s>")
                sent.insert(len(sent),"</s>")
                sent.insert(len(sent),"</s>")
        #Word tokenization for training sentences:
        train_words=[]
        for i in range(0,len(train)):
                for j in range(0,len(train[i])):
                        train_words.append(train[i][j])
        unigram_count=Counter(train_words)
    
        bigrams=[]
        bigram=nltk.bigrams(train_words)
        for bg in bigram:
                bigrams.append(bg)
        bigram_count=Counter(bigrams)
    
        bigram_count.pop(('</s>','<s>'))  # removing the unnencessary bigrams
    
        trigrams=[]
        trigram=nltk.trigrams(train_words)
        for tg in trigram:
                trigrams.append(tg)
        trigram_count=Counter(trigrams)
        trigram_count.pop(('</s>','</s>','<s>'))
        trigram_count.pop(('</s>','<s>','<s>')) # removing the unnencessary trigrams
    
        while True:

                word=[ 'The','He','This','His','Her','Our','An'] # possible starting words

                temp=np.random.choice(len(word),1)
                word=word[temp[0]]
                sentence=[]
                for i in range(0,5):
                        word_list=[]
                        for keys in trigram_count:
                                if keys[0]==word:
                                        word_list.append(keys[1:3])    
                        if len(word_list)== 0:
                                word='the'
                        word_list=[]
                        for keys in trigram_count:
                                if keys[0]==word:
                                        word_list.append(keys[1:3])    
            #print(word)
                        while True:
                                x=np.random.choice(len(word_list),1)
                                word_trigram=(word,word_list[x[0]][0],word_list[x[0]][1])
                #print(word_trigram)
                                prob = trigram_count[word_trigram]/unigram_count[word]
                #print(prob)
                                rv=np.random.random_sample()
                                if rv<prob:
                                        sentence.append(word)
                                        sentence.append(word_trigram[1])
                                        word=word_trigram[2]
                                        break


                if sentence[9]=='.':
                        break;
        print(sentence)


# In[21]:



# For printing the perplexity results
print('Results for S1:')
test_train_selection(1)

print('Results for S2:')
test_train_selection(2)

print('Results for S3:')
test_train_selection(3)

print('Results for S4:')
test_train_selection(4)


# In[27]:


# For random Sentence generation
print(' ')
print('The random sentence generated is:')
print(' ')
random_sent(list(brown.sents()))

