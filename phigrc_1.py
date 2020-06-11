import spacy
import pandas as pd
import random
import re
import math

print("loading nlp model ")
nlp= spacy.load("en_core_web_lg")

print("loaded nlp model ")


from gensim   import corpora
import gensim
import time
from gensim import corpora , models as gmp
from operator import itemgetter
import psutil

import copy
import pickle
import nltk
from nltk.corpus import stopwords
import re
from multiprocessing import Process

import sys
import csv   
import urllib
import json
import math
import numpy as np
import scipy
import spacy

import glob
from collections import Counter 
from gensim import models as gm
from pathlib import Path
import os
import time
from subprocess import call
from numpy.random import seed
import random

sys.setrecursionlimit(1000000)
seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED']='0'  
try :
        os.environ['OPENBLAS_NUM_THREADS']="1"
        import numpy
finally:
       if 'OPENBLAS_NUM_THREADS' in os.environ:
              del os.environ['OPENBLAS_NUM_THREADS']

tagr=spacy.load('en',disable=['ner','parser'])

start_t=time.time()
is_ascii=lambda s:len(s)==len(s.encode())

def  print_timstamp(t=1):
         if(t==1):
               print(str(time.asctime()))
               print("\t"+time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime()))
         return time.time()

def diff_timstamp():
         print("\ttim diff from start"+str(start_t-print_timstamp(0)))
         print_timstamp()
                          
def rmov_nulls(s):
         ia=s
         y=copy.deepcopy(ia)
         for i in range(len(ia)):
             if('' in y): 
                      y.remove('')
             else:
                 pattrn_sp=re.compile(r'^(\s|\*|\.|_|-|\+|\d)+$')
                 pattrn_match=pattrn_sp.match(ia[i])
                 if ia[i] in y and pattrn_match is not None:
                     y.remove(ia[i])
                 else:
                     pattrn_sp=re.compile(r'^(\-|\_|\=)')
                     pattrn_match=pattrn_sp.match(ia[i])
                     if ia[i] in y and pattrn_match is not None:
                         y.remove(ia[i])
                     else:
                         pattrn_sp=re.compile(r'^(\w|\.|\-|_|\+|\/|\$|\d)$')
                         pattrn_match=pattrn_sp.match(ia[i])  
                         if ia[i] in y and pattrn_match is not None:
                             y.remove(ia[i])                        
         return y    
 


def lmma_cln(strings):
         strings1=[]
         strings_1=[]
         strings_2=[]
         strings_lm=[]
         uyp_=[]
         str2_i=[]
         str1_i=[]
         i=0
         for string in strings:
          
            if(not re.search(r'(\_|\-|\.)',string)):
               strings_2.append(string)
               str2_i.append(i)

            if(re.search(r'(\_|\-|\.)',string)):
               
               strings_1.append(string)     
 
               str1_i.append(i)        

            i=i+1

         words_stagr=tagr(' '.join(strings_2))        

         strings_lm=[wd.lemma_ for wd in words_stagr]                        

         strings_1.reverse()

         strings_lm.reverse()
       
         i=0

         for string in strings:

            if(i in str1_i):
  
               if(len(strings_1)>0):   
                       
                  strings1.append(strings_1.pop())

            if(i in str2_i):
                                                         
               if(len(strings_lm)>0):

                  strings1.append(strings_lm.pop())
         
            i=i+1
         
         return strings1
                      
def  data_process(input_df,again=True ):
                  
         
         nltk.download('stopwords')
         stop_words=stopwords.words('english') 
                 
         #data profil  
         print("data procssing")
         diff_timstamp()
      
         print("\tdim:"+str(len(input_df)))
         input_df['words']=None
         
         word_count=0        
         char_count=0
         input_df['t-hash']=None
         df_list=[]          
         if(again==True):
             for ind,valu in input_df.iterrows():
             
                 if(ind%100==0):
                    print(ind,end='')        
                    
                 text=valu['text'].lower()
                 split_snt=re.split(r'(\.\s+|\?\s+|\!\s+|\n|\\n)',text)
                 snts=rmov_nulls(split_snt)
                 
                 tokns=[]
                 for snt in snts:
                     dp_df={}
                     
                     if(ind%250==0):
                          print("ind:+"+str(ind))
                          print("ra:"+str(snt))
                     split_q1=snt.split(' ')
                 
                     split_t=re.split(r'[,\\\(\)\;\s+\t\~&\=\%\>\<\`\{\}\|\#\*\+\^]',snt)
                     if(ind%250==0):
                          print("isp:"+str(split_t))
                     
                     """split_t=[t for t1 in split_t for t in t1]
                     ""split_t=rmov_nulls(split_t)
                     """
                     if(ind%250==0):
                          print("aftr:rmov1:"+str(split_t))
                     
                     """split_t=[re.sub(r'(\'|\")','',t) for t in split_t]
                     split_t=[re.sub(r'(\(|\[|\)|\]|\!|\+|\=|\\|\*)',' ',t) for t in split_t]
                     """
                     if(ind%250==0):
                          print("aftr :sub:"+str(split_t))
                     
                     """split_t=rmov_nulls(split_t)
                     """
                     split_t=[t for t in split_t if is_ascii(t)]
                     """split_t=[re.sub(r'(\.|\-|_)$','',t) for t in split_t]
                     """
                     split_t1=[t for t in split_t if re.search(r'(\.|\-|_)',t)]
                     
                     split_t2=[t for t in split_t if not re.search(r'(\.|\-|\_)',t)]
                       
                     split_lmma= lmma_cln(split_t)                              
                     
                     split_lmma=split_t
                     split_lmma=[d for d in split_lmma if d !='-PRON-']
                     """split_lmma=rmov_nulls(split_lmma)
                     """
                     split_lmma=[t for t in split_lmma if len(t)>2 or t in ['no','go','if','qa']]
                     split_lmma=[re.sub('\s+',' ' , tt1) for tt1 in split_lmma]
                     split_lmma=[re.sub('\.$','',tt1) for tt1 in split_lmma]
                     if(ind%250==0):
                          print("aftr lmma:"+str(split_lmma))

                     
                     split_lmma=[t for t in split_lmma if t not in stop_words]
                     """split_lmma= lmma_cln(split_t)
                     """
                     if(ind%250==0):
                        
                           print("aftr stpw:"+str(split_lmma))
 
                           if(ind%100==0):

                                input_df.to_csv('Procssd'+str(random.randint(10,100))+'.csv')
                                              
                     
                     tokns=[]
                 for i in range(len(split_lmma)):
                         worda=split_lmma[i]
                         tokns.append(worda)
                    
                 valu['words']=split_lmma
                 valu['t-hash']=str(tokns)
                 valu['s-hash']=''
   
                 if(len(tokns)>0):

                    valu['s-hash']=str(list(set(tokns)).sort())
                                                                
                 word_count=word_count+len(tokns)
                      
                 char_count1=sum([len(sq) for sq in tokns])
                 char_count=char_count+char_count1
                 df_list.append(valu)
             print("\nword_count:"+str(word_count))

             print("char_count:"+str(char_count))
                                     
 
         hash_df=pd.DataFrame(df_list,columns=df_list[0].keys()) 
         uni_dict={}
         for i,v in hash_df.iterrows():
            for word in v['words']:
                if word in uni_dict:
                    uni_dict[word]=uni_dict[word]+1
                if word not in uni_dict:
                    uni_dict[word]=1
         
         hash_df['lwords']=None
         word_df={}
         word_dl=[]
         uni_dict_f={}
         for i,v in hash_df.iterrows():
                for word in v['words']:
                    wrd_l=[] 
                    if uni_dict[word]>5:
                             wrd_l.append(word)

                             if word in uni_dict_f:
                                     uni_dict_f[word]=uni_dict_f[word]+1
                             if word not in uni_dict_f:
                                     uni_dict_f[word]=1
                                     
                v['lwords']=wrd_l
                word_dl.append(v)
         
         word_df=pd.DataFrame(word_dl,columns=word_dl[0].keys())
         
         """['t-hash','words','list'])
             """
         print('filtrd word frq'+str(len(uni_dict_f.keys())))
         diff_timstamp()    
         
         return word_df

def lda(hash_df, again=True,scoring=False):
         
         diff_timstamp() 
         
         if(again==True):
             h_df={}
             h_LDA_df=hash_df
             if( len(h_LDA_df)<1):
                  h_LDA_df=hash_df
                  h_df=h_LDA_df
                  
             print("LDA building start")
             #print_timstamp()
             topic_word={}
             t_wldfjk=[]
             for l,r in h_LDA_df.iterrows():
                 split_co=re.split(r',',str(r['lwords']))
                 split_co=[re.sub(r'[\'\[\]]','',i) for i in split_co]
                 t_wldfjk.append(split_co)              
             t_array=t_wldfjk
             LDA_dict=corpora.Dictionary(t_array)

             corpus=[LDA_dict.doc2bow(i) for i in t_array]
             print("LDA building")
             diff_timstamp()
             LDA_fil_nam_dictionary='LDA_dict.p'
             LDA_fil_nam_topics='LDA_Topics'+str(random.randint(10,100))+'.csv'
             dict_fil_nam= LDA_fil_nam_dictionary
             dict_fil=open(dict_fil_nam,"wb")
             pickle.dump(LDA_dict,dict_fil)
             dict_fil.close()
             num_topics=None
             
             if num_topics is None:
                num_topics=math.floor(len(h_LDA_df)/(100))
             LDA_modl=gensim.models.LdaModel(corpus,num_topics= 25,id2word=LDA_dict,passes=2000)
             LDA_modl.save("LDA_fil_nam_"+str(random.randint(10,100))+".modl")

             topics=LDA_modl.print_topics(-1)
             
             """if(rounding==True):
                 topic_roof=0
                 if(round_cnt>0):
                    r_topic_df=pd.read_csv(LDA_fil_nam_r_topics)
                    topic_roof=int(r_topic_df.iloc[len(r_topic_df)-1][0])+1
                    
                 with open(LDA_fil_nam_r_topics,'a+',encoding='utf-8') as csv_topics:
                    for i in topics:             
                       if(type(i[0]) is int):
                             csv_topics.write(str(int(i[0])+topic_roof)+","+str(i[1]))
                       if(round_cnt==0):
                          if (type(i[0]) is not int):
                             csv_topics.write(i[0]+","+i[1])        
             """
             with open(LDA_fil_nam_topics,'w',encoding='utf-8') as  csv_topic:
                 writr=csv.writer(csv_topic,delimiter=',')
                 for i in topics:
                     writr.writerow(i)
             topics_mapping=LDA_modl[corpus]
             print("topic matching probability binns")

             prob_dict={}
             topic_dist_list=[]
             for i in topics_mapping:
                
                 l=sorted(i,key=lambda s:s[1],reverse=True)
                 topic_dist_list.append(l)
                 prob=1+math.floor(float(l[0][1])/0.1)
                 prob=str(prob)
                 if prob not in prob_dict:
                    prob_dict[prob]=1
                 if prob in prob_dict:
                    prob_dict[prob]=prob_dict[prob]+1

             for d in range(10,-1,1):
                ad=str(d)              
                print("probability match "+str(ad)+" row count:"+str(prob_dict[ad]))
          
             topic_map_df=pd.DataFrame(topic_dist_list)
             topic_map_df=topic_map_df[topic_map_df.columns[0:2]]
             topic_map_df.columns=['topic1','topic2']
             topic_map_df[['Topic 1 match', 'Topic 1 probability']]=topic_map_df['topic1'].apply(pd.Series)
             topic_map_df[['Topic 2 match','Topic 2 probability']]=topic_map_df['topic2'].apply(pd.Series)
             topic_dist_df=topic_map_df[['Topic 1 match','Topic 1 probability','Topic 2 match','Topic 2 probability']]   
             
             for k in topic_dist_df:
                 h_LDA_df[k]=topic_dist_df[k]
             h_LDA_df.to_csv("Topic Modelling"+str(random.randint(10,100))+".csv")         
             """LDA_df=hash_df    
             #self.LDA_df['t-hash']=self.LDA_df['words']
             LDA_df=LDA_df.join(h_LDA_df.set_index('t-hash'),on='t-hash',lsuffix="l_",rsuffix='r_')
             LDA_df.drop(['fr'],axis=1)
                                              
             LDA_df.to_csv("Topic Modelling.csv")
             """
             diff_timstamp()
 
 if __name__=='__main__':        
     print('initialisation stage ')
     print(" data loading and sanity checks ")
     config_file_handle=open('config.json',"r",encoding='utf-8')
    
     assert config_file_handle, ' Could not open config File'
    
     config= json.load(config_file_handle)
 
     print('loading input dataset')
    
     input_data=pd.read_csv(config["DataURL"],keep_default_na=False,encoding='latin-1')  
     dataframe= input_data
	 
     assert input_data," could not open input file "
	 
     dataframe['text']=dataframe['Section_Text']
     """datafram_1=data_process(dataframe)
     lda(datafram_1)
     """
     simil_dict={}
     pos_l=[] 
     pos_df={}
     sntncs=[]
     pos_dict={}
     corpus=[]

     for i,v in dataframe.iterrows():
         string=v['Section_Text']
         usc='u.s.c.'
  
         if(re.search(r'(\w\.\w\.\w\.)',string)):

             str_match=re.search(r'(\w\.\w\.\w\.)',string)

             usc=str_match.groups()[0]

             string=string.replace(str_match.groups()[0],'USC')
      
       
             strs=re.split(r'\.\s*',string)
             w=0
	     i=0
		
             for str_ in strs:
                 str_=str_.replace('USC',usc)
                 nlp_doc=nlp(str_)
                 pos_dict={}
                 str_pos=''
                 for token in nlp_doc:
                     '''print(token.text+"pos:"+token.pos_)'''
                     str_pos=str_pos+" "+str(token.text)+"/"+str(token.pos_)+ " "
                     """print(str_pos)
                     """
                 if(len(str_)>2):
                     pos_dict['Cite']=i
                     pos_dict['text']=str_
                     pos_dict['Text with POS']=str_pos
                     pos_l.append(pos_dict)
                     sntncs.append(str_)
            
            
                 print(' i : ' + str(i))
             df=pd.DataFrame(pos_l,columns= ['Cite','text','Text with POS'])
             df.to_csv('POStoCite _all '+str(random.randint(0,i))+'.csv')
     
     datafram_1=data_process(df)
     lda(datafram_1)
  
     simil_df={}
     simil_l=[]
     inti=0

     for i in sntncs:

         print(str(inti)+"snt similarity")
         inti=inti+1
         intj=0
         for j in sntncs:
             if i!=j :
                 if(intj%100==0):
                     print(str(intj)+"snt similarity")

             intj=intj+1
        
             simil_dict={}
             nlp_doc=nlp(i)
             nlp_doc1=nlp(j)
             similarit=nlp_doc.similarity(nlp_doc1)

             simil_dict['snt1']=i
             simil_dict['snt2']=j
             simil_dict['similarity']=similarit
        
             similarit=int(math.floor(similarit*100))

             if similarit in simil_dict:
                 simil_dict[similarit]=simil_dict[similarit]+1
             if similarit not in simil_dict:
                 simil_dict[similarit]=1    


             simil_l.append(simil_dict)
             if intj%1000==0:
                 simil_df=pd.DataFrame(simil_l,columns=['snt1','snt2','similarity'])        
                 simil_df.to_csv('Sentence_similarity'+str(inti)+'.csv')               
