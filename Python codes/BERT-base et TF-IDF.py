#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy
from spacy.lang.en import STOP_WORDS
import numpy as np
import math
import pandas as pd
from nltk.stem import WordNetLemmatizer
import re
import nltk
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer


# In[3]:


# Dans cette partie, je lis les fichiers .txt décrits ci-dessus et je les stocke dans les listes python

with open("D:\\WISD-S3\\mon stage pfe\\mon stage pfe\\code Billy Ngaba\\STAGE_M2_SSD\\variables_sources.txt", "r",encoding='utf-8') as file:
    lines_var_src=file.read().splitlines()
    file.close()

lines_var_src1 = []
for i in range(len(lines_var_src)):
    lines_var_src1.append(re.sub('_',' ', lines_var_src[i])) # remplacement des '_' par des espaces

with open('D:\\WISD-S3\\mon stage pfe\\mon stage pfe\\code Billy Ngaba\\STAGE_M2_SSD\\descriptions_sources.txt', "r",encoding='utf-8') as file:
    lines_des_src = file.read().splitlines()
    file.close()


with open('D:\\WISD-S3\\mon stage pfe\\mon stage pfe\\code Billy Ngaba\\STAGE_M2_SSD\\variables_candidates.txt', "r",encoding='utf-8') as file:
    lines_var_cand = file.read().splitlines()
    file.close()

lines_var_cand1 = []
for i in range(len(lines_var_cand)):
    lines_var_cand1.append(re.sub('_',' ', lines_var_cand[i])) # remplacement des '_' par des espaces



with open('D:\\WISD-S3\\mon stage pfe\\mon stage pfe\\code Billy Ngaba\\STAGE_M2_SSD\\descriptions_candidates.txt', "r",encoding='utf-8') as file:
    lines_des_cand = file.read().splitlines()
    file.close()
    
#with open('D:\\WISD-S3\\mon stage pfe\\mon stage pfe\\code Billy Ngaba\\projet\\plantedeservices-master\\Code\\mes executions\\fichier text_mining_complement_approche_contextuelle\\complement_approche_contextuelle.txt', "r",encoding='utf-8') as file:
#    lines_contexte = file.read().splitlines()
#    file.close()


# In[4]:


from pathlib import Path

text1 = Path('D:\\WISD-S3\\mon stage pfe\\mon stage pfe\\code Billy Ngaba\\projet\\plantedeservices-master\\Code\\mes executions\\fichier text_mining_complement_approche_contextuelle\\Docs_convertis\\Akemo-Wortman-2.txt',encoding='utf-8').read_text()
text2 = Path('D:\\WISD-S3\\mon stage pfe\\mon stage pfe\\code Billy Ngaba\\projet\\plantedeservices-master\\Code\\mes executions\\fichier text_mining_complement_approche_contextuelle\\Docs_convertis\Aude_merged.txt').read_text(encoding='utf-8')
text3 = Path('D:\\WISD-S3\\mon stage pfe\\mon stage pfe\\code Billy Ngaba\\projet\\plantedeservices-master\\Code\\mes executions\\fichier text_mining_complement_approche_contextuelle\\Docs_convertis\Baraibar-Gfeller-10.txt').read_text(encoding='utf-8')
text4 = Path('D:\\WISD-S3\\mon stage pfe\\mon stage pfe\\code Billy Ngaba\\projet\\plantedeservices-master\\Code\\mes executions\\fichier text_mining_complement_approche_contextuelle\\Docs_convertis\Beillouin_merged.txt').read_text(encoding='utf-8')
text5 = Path('D:\\WISD-S3\\mon stage pfe\\mon stage pfe\\code Billy Ngaba\\projet\\plantedeservices-master\\Code\\mes executions\\fichier text_mining_complement_approche_contextuelle\\Docs_convertis\Berry2009_merged.txt').read_text(encoding='utf-8')
text6 = Path('D:\\WISD-S3\\mon stage pfe\\mon stage pfe\\code Billy Ngaba\\projet\\plantedeservices-master\\Code\\mes executions\\fichier text_mining_complement_approche_contextuelle\\Docs_convertis\Cajas2019_These_merged.txt').read_text(encoding='utf-8')
text7 = Path('D:\\WISD-S3\\mon stage pfe\\mon stage pfe\\code Billy Ngaba\\projet\\plantedeservices-master\\Code\\mes executions\\fichier text_mining_complement_approche_contextuelle\\Docs_convertis\Christina2021.txt').read_text(encoding='utf-8')
text8 = Path('D:\\WISD-S3\\mon stage pfe\\mon stage pfe\\code Billy Ngaba\\projet\\plantedeservices-master\\Code\\mes executions\\fichier text_mining_complement_approche_contextuelle\\Docs_convertis\Damien_merged.txt').read_text(encoding='utf-8')
text9 = Path('D:\\WISD-S3\\mon stage pfe\\mon stage pfe\\code Billy Ngaba\\projet\\plantedeservices-master\\Code\\mes executions\\fichier text_mining_complement_approche_contextuelle\\Docs_convertis\Damour_merged.txt').read_text(encoding='utf-8')
text10 = Path('D:\\WISD-S3\\mon stage pfe\\mon stage pfe\\code Billy Ngaba\\projet\\plantedeservices-master\\Code\\mes executions\\fichier text_mining_complement_approche_contextuelle\\Docs_convertis\Hajjar-Malezieux10.txt').read_text(encoding='utf-8')
text11 = Path('D:\\WISD-S3\\mon stage pfe\\mon stage pfe\\code Billy Ngaba\\projet\\plantedeservices-master\\Code\\mes executions\\fichier text_mining_complement_approche_contextuelle\\Docs_convertis\muhammad-Tribouillois-10.txt',encoding='utf-8').read_text()
text12 = Path('D:\\WISD-S3\\mon stage pfe\\mon stage pfe\\code Billy Ngaba\\projet\\plantedeservices-master\\Code\\mes executions\\fichier text_mining_complement_approche_contextuelle\\Docs_convertis\Rakotomanga_ASD_merged.txt').read_text(encoding='utf-8')
text13 = Path('D:\\WISD-S3\\mon stage pfe\\mon stage pfe\\code Billy Ngaba\\projet\\plantedeservices-master\\Code\\mes executions\\fichier text_mining_complement_approche_contextuelle\\Docs_convertis\Ratnadass_2020_merged.txt').read_text(encoding='utf-8')
text14 = Path('D:\\WISD-S3\\mon stage pfe\\mon stage pfe\\code Billy Ngaba\\projet\\plantedeservices-master\\Code\\mes executions\\fichier text_mining_complement_approche_contextuelle\\Docs_convertis\Sobia_merged.txt').read_text(encoding='utf-8')
text15 = Path('D:\\WISD-S3\\mon stage pfe\\mon stage pfe\\code Billy Ngaba\\projet\\plantedeservices-master\\Code\\mes executions\\fichier text_mining_complement_approche_contextuelle\\Docs_convertis\Weeds of tropical rainfed cropping systems.txt').read_text(encoding='utf-8')


# Je regroupe tous les articles 'nettoyés' en un seul texte
#text=" ".join([text1,text2,text3,text4,text5,text6,text7,text8,text9,text10,text11,text12,text13,text14,text15])
text =[]
text.append(text1)
text.append(text2)
text.append(text3)
text.append(text4)
text.append(text5)
text.append(text6)
text.append(text7)
text.append(text8)
text.append(text9)
text.append(text10)
text.append(text11)
text.append(text12)
text.append(text13)
text.append(text14)
text.append(text15)



# Supprimer les URLs
contexte_without_urls = re.sub(r'http\S+|www\S+|https\S+', '', str(text))

# Supprimer les caractères "\n" successifs
contexte_cleaned = re.sub(r'\n\n+', '', contexte_without_urls)
contexte = contexte_cleaned.split('\n')

 


# In[5]:


############################################### prétraitement des descriptions des variables ##################################################### 

def clean_text(texte):
    # Supprimer les nombres
    texte = re.sub(r'\d+', '', texte)
    
    # Supprimer les parenthèses et leur contenu
    texte = re.sub(r'\([^()]*\)', '', texte)
    
    # Supprimer les autres caractères spéciaux
    texte = re.sub(r'[^a-zA-Z0-9\s]', '', texte)
    
    # Convertir en minuscules
    texte = texte.lower()
    
    # Retourner le texte nettoyée
    return texte

lines_des_cand_pre = []
for line in lines_des_cand:
    lines_des_cand_pre.append(clean_text(line))
    
lines_des_src_pre = []
for line in lines_des_src:
    lines_des_src_pre.append(clean_text(line))
    


    
    
######################################################################

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def remove_stopwords(texte):
    stop_words = set(stopwords.words('english'))  # Choisissez la langue appropriée
    words = texte.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


lines_des_cand_pre2 = []
for line in lines_des_cand_pre:
    lines_des_cand_pre2.append(remove_stopwords(line))
    

lines_des_src_pre2 = []
for line in lines_des_src_pre:
    lines_des_src_pre2.append(remove_stopwords(line))
    
    
    
############################################################

import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

def lemmatize(texte):
    lemmatizer = WordNetLemmatizer()
    words = texte.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)


lines_des_src_pre3 = []
for line in lines_des_src_pre2:
    lines_des_src_pre3.append(lemmatize(line))


    
lines_des_cand_pre3 = []
for line in lines_des_cand_pre2:
    lines_des_cand_pre3.append(lemmatize(line))
    



    
##############################################################

import string

def remove_punctuation(texte):
    translator = str.maketrans('', '', string.punctuation)
    texte = texte.translate(translator)
    return texte


lines_des_src_pre4 = []
for line in lines_des_src_pre3:
    lines_des_src_pre4.append(remove_punctuation(line))


lines_des_cand_pre4 = []
for line in lines_des_cand_pre3:
    lines_des_cand_pre4.append(remove_punctuation(line))

    
    
##################################################################

from nltk.corpus import wordnet

def replace_synonyms(texte):
    words = texte.split()
    replaced_words = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            replaced_words.append(synonyms[0].lemmas()[0].name())
        else:
            replaced_words.append(word)
    return ' '.join(replaced_words)



lines_des_src_pre5 = []
for line in lines_des_src_pre4:
    lines_des_src_pre5.append(replace_synonyms(line))
    

lines_des_cand_pre5 = []
for line in lines_des_cand_pre4:
    lines_des_cand_pre5.append(replace_synonyms(line))
    








    
###################################################################### prétraitement des noms des variables. ######################

 


lines_var_cand_pre = []
for line in lines_var_cand:
    lines_var_cand_pre.append(clean_text(line))
    
lines_var_src_pre = []
for line in lines_var_src:
    lines_var_src_pre.append(clean_text(line))



lines_var_cand_pre2 = []
for line in lines_var_cand_pre:
    lines_var_cand_pre2.append(remove_stopwords(line))
    

lines_var_src_pre2 = []
for line in lines_var_src_pre:
    lines_var_src_pre2.append(remove_stopwords(line))


lines_var_src_pre3 = []
for line in lines_var_src_pre2:
    lines_var_src_pre3.append(lemmatize(line))


    
lines_var_cand_pre3 = []
for line in lines_var_cand_pre2:
    lines_var_cand_pre3.append(lemmatize(line))



    
    
    
###################################################################### prétraitement des articles pour le contexte


contexte_pre = []
for line in contexte:
    contexte_pre.append(clean_text(line))
    


contexte_pre2 = []
for line in contexte_pre:
    contexte_pre2.append(remove_stopwords(line))
    

contexte_pre3 = []
for line in contexte_pre2:
    contexte_pre3.append(lemmatize(line))



contexte_pre4 = []
for line in contexte_pre3:
    contexte_pre4.append(remove_punctuation(line))



contexte_pre5 = []
for line in contexte_pre4:
    contexte_pre5.append(replace_synonyms(line))
    





# In[6]:


import torch
from transformers import BertTokenizer, BertModel, BertConfig

################################################
config = BertConfig(num_hidden_layers=2)

# Charger le modèle BERT pré-entraîné
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',config=config)
model = BertModel.from_pretrained('bert-base-uncased',config=config)

# Prétraitement des descriptions des variables sources
src_inputs_leven = tokenizer(lines_var_src_pre3, padding=True, truncation=True, return_tensors="pt")
src_outputs_leven = model(**src_inputs_leven)

# Prétraitement des descriptions des variables candidates
cand_inputs_leven = tokenizer(lines_var_cand_pre3, padding=True, truncation=True, return_tensors="pt")
cand_outputs_leven = model(**cand_inputs_leven)

# Calculer les similarités entre les embeddings des descriptions
similarity_scores_leven = torch.cosine_similarity(src_outputs_leven.last_hidden_state.mean(dim=1).unsqueeze(1),
                                            cand_outputs_leven.last_hidden_state.mean(dim=1).unsqueeze(0), dim=2)


# In[7]:


############################################################## TFIDF #######################################################


# In[8]:


#################################

# Vocabulaire pré-traité
vocab_sans_ponctuations = lines_des_cand_pre5 + lines_var_cand_pre3 + contexte_pre5 + lines_des_src_pre5 + lines_var_src_pre3#+ lines_des_src


# In[9]:


vectorizer = TfidfVectorizer(stop_words='english') 
esp_vec = vectorizer.fit(vocab_sans_ponctuations)
des_src_vect = esp_vec.transform(lines_des_src_pre5)
#var_cand_vect = esp_vec.transform(lines_var_cand)
des_cand_vect = esp_vec.transform(lines_des_cand_pre5)


# In[10]:


## Evaluation
def evaluation(numero,x, rang,lines_var_src, lines_var_cand ):
    i = numero
    cosinus1 = [] # avec les des. cand
    lev = []
    combi = []
    # x dans [0,1]
    for k in range(len(lines_var_cand)):
        cosinus1.append(cosine_similarity(des_src_vect[i],des_cand_vect[k])[0][0])
        lev.append(similarity_scores_leven[i][k].item())
        combi.append(x*cosine_similarity(des_src_vect[i],des_cand_vect[k])[0][0] + (1-x)*similarity_scores_leven[i][k].item())
    ##
    df_lev = pd.DataFrame({'var_src':[lines_var_src[i]]*len(lines_var_cand),'var. cand. avec les meilleurs scores sur cosinus(nom de variables)': lines_var_cand,'lev': lev })
    df_lev = df_lev.sort_values(by='lev', ascending=False)
    df_lev.reset_index(drop=True, inplace=True) 
    ##
    df_cos = pd.DataFrame({'var_src':[lines_var_src[i]]*len(lines_var_cand),'var. cand. avec les meilleurs scores sur cosinus(description de variables)': lines_var_cand, 'cos des cand': cosinus1})
    df_cos = df_cos.sort_values(by = 'cos des cand', ascending=False)
    df_cos.reset_index(drop=True, inplace=True)
    ##
    df_combi = pd.DataFrame({'var_src':[lines_var_src[i]]*len(lines_var_cand),'var. cand. avec les meilleurs scores sur la combinaison': lines_var_cand,'combi': combi})
    df_combi = df_combi.sort_values(by = 'combi', ascending=False)
    df_combi.reset_index(drop=True, inplace=True)
    ##
    df_eval = pd.concat([df_lev,df_cos,df_combi], axis = 1)
    return df_eval.head(rang)


# In[11]:


############################################################ fin TFIDF ######################################################


# In[12]:


####################### Tableaux des évaluations

correspondances = pd.read_excel("D:\\WISD-S3\\mon stage pfe\\mon stage pfe\\code Billy Ngaba\\STAGE_M2_SSD\\Correspondances.xlsx")
varSrcList = correspondances["Variable source"]


# In[13]:


import pandas as pd

TableauGlobLev = pd.DataFrame({
        'Variable source': varSrcList,
        'rang 1': [0]*len(lines_var_src),
        'rang 3': [0]*len(lines_var_src),
        'rang 5': [0]*len(lines_var_src),
        'rang 10': [0]*len(lines_var_src)
    })


## Lev
    

liste_varScr_precision_inf_10_lev = []
liste_varScr_precision_inf_5_lev = []
liste_varScr_precision_inf_3_lev = []

num_var_src = 0

for i in varSrcList:
    df_eval10 = evaluation(num_var_src, 0.25, 10, lines_var_src1, lines_var_cand1)
    df_eval5 = evaluation(num_var_src, 0.25, 5, lines_var_src1, lines_var_cand1)
    df_eval3 = evaluation(num_var_src, 0.25, 3, lines_var_src1, lines_var_cand1)
    df_eval1 = evaluation(num_var_src, 0.25, 1, lines_var_src1, lines_var_cand1)
    Valeur = correspondances.loc[correspondances['Variable source'] == i]['Variable correspondante'].values[0]

    if Valeur in df_eval1["var. cand. avec les meilleurs scores sur cosinus(nom de variables)"].values:
        TableauGlobLev.loc[TableauGlobLev['Variable source'] == i, 'rang 10'] = 1
        TableauGlobLev.loc[TableauGlobLev['Variable source'] == i, 'rang 5'] = 1
        TableauGlobLev.loc[TableauGlobLev['Variable source'] == i, 'rang 3'] = 1
        TableauGlobLev.loc[TableauGlobLev['Variable source'] == i, 'rang 1'] = 1
    else:
        if Valeur in df_eval3["var. cand. avec les meilleurs scores sur cosinus(nom de variables)"].values:
            TableauGlobLev.loc[TableauGlobLev['Variable source'] == i, 'rang 3'] = 1
            TableauGlobLev.loc[TableauGlobLev['Variable source'] == i, 'rang 5'] = 1
            TableauGlobLev.loc[TableauGlobLev['Variable source'] == i, 'rang 10'] = 1
        else:
            liste_varScr_precision_inf_3_lev.append(i)
            if Valeur in df_eval5["var. cand. avec les meilleurs scores sur cosinus(nom de variables)"].values:
                TableauGlobLev.loc[TableauGlobLev['Variable source'] == i, 'rang 5'] = 1
                TableauGlobLev.loc[TableauGlobLev['Variable source'] == i, 'rang 10'] = 1
            else:
                liste_varScr_precision_inf_5_lev.append(i)
                if Valeur in df_eval10["var. cand. avec les meilleurs scores sur cosinus(nom de variables)"].values: 
                    TableauGlobLev.loc[TableauGlobLev['Variable source'] == i, 'rang 10'] = 1
                else:
                    liste_varScr_precision_inf_10_lev.append(i)
    num_var_src += 1
    


# In[14]:


import pandas as pd

TableauGlobCos = pd.DataFrame({
        'Variable source': varSrcList,
        'rang 1': [0]*len(lines_var_src),
        'rang 3': [0]*len(lines_var_src),
        'rang 5': [0]*len(lines_var_src),
        'rang 10': [0]*len(lines_var_src)
    })

## cosinus
     

liste_varScr_precision_inf_10_cos = []
liste_varScr_precision_inf_5_cos = []
liste_varScr_precision_inf_3_cos = []

num_var_src = 0

for i in varSrcList:
    df_eval10 = evaluation(num_var_src, 0.25, 10, lines_var_src1, lines_var_cand1)
    df_eval5 = evaluation(num_var_src, 0.25, 5, lines_var_src1, lines_var_cand1)
    df_eval3 = evaluation(num_var_src, 0.25, 3, lines_var_src1, lines_var_cand1)
    df_eval1 = evaluation(num_var_src, 0.25, 1, lines_var_src1, lines_var_cand1)
    Valeur = correspondances.loc[correspondances['Variable source'] == i]['Variable correspondante'].values[0]

    if Valeur in df_eval1["var. cand. avec les meilleurs scores sur cosinus(description de variables)"].values:
        TableauGlobCos.loc[TableauGlobCos['Variable source'] == i, 'rang 1'] = 1
        TableauGlobCos.loc[TableauGlobCos['Variable source'] == i, 'rang 3'] = 1
        TableauGlobCos.loc[TableauGlobCos['Variable source'] == i, 'rang 5'] = 1
        TableauGlobCos.loc[TableauGlobCos['Variable source'] == i, 'rang 10'] = 1
    else:
        if Valeur in df_eval3["var. cand. avec les meilleurs scores sur cosinus(description de variables)"].values:
            TableauGlobCos.loc[TableauGlobCos['Variable source'] == i, 'rang 3'] = 1
            TableauGlobCos.loc[TableauGlobCos['Variable source'] == i, 'rang 5'] = 1
            TableauGlobCos.loc[TableauGlobCos['Variable source'] == i, 'rang 10'] = 1
        else:
            liste_varScr_precision_inf_3_cos.append(i)
            if Valeur in df_eval5["var. cand. avec les meilleurs scores sur cosinus(description de variables)"].values:
                TableauGlobCos.loc[TableauGlobCos['Variable source'] == i, 'rang 5'] = 1
                TableauGlobCos.loc[TableauGlobCos['Variable source'] == i, 'rang 10'] = 1
            else:
                liste_varScr_precision_inf_5_cos.append(i)
                if Valeur in df_eval10["var. cand. avec les meilleurs scores sur cosinus(description de variables)"].values:
                    TableauGlobCos.loc[TableauGlobCos['Variable source'] == i, 'rang 10'] = 1
                else:
                    liste_varScr_precision_inf_10_cos.append(i)
    num_var_src += 1


# In[15]:


TableauGlobComb = pd.DataFrame({'Variable source':varSrcList,'rang 1': [0]*len(lines_var_src),'rang 3': [0]*len(lines_var_src),'rang 5': [0]*len(lines_var_src),'rang 10': [0]*len(lines_var_src) })


###### combi 
    
num_var_src = 0
for i in varSrcList: 
    df_eval10 = evaluation(num_var_src,0.25, 10,lines_var_src1, lines_var_cand1 )
    df_eval5 = evaluation(num_var_src,0.25, 5,lines_var_src1, lines_var_cand1 )
    df_eval3 = evaluation(num_var_src,0.25,3,lines_var_src1, lines_var_cand1 )
    df_eval1 = evaluation(num_var_src,0.25, 1,lines_var_src1, lines_var_cand1 )
    Valeur = correspondances.loc[correspondances['Variable source'] == i]['Variable correspondante'].values[0]
    if(Valeur in df_eval1["var. cand. avec les meilleurs scores sur la combinaison"].values):
        TableauGlobComb.loc[TableauGlobComb['Variable source'] == i,'rang 10'] = 1
        TableauGlobComb.loc[TableauGlobComb['Variable source'] == i,'rang 5'] = 1
        TableauGlobComb.loc[TableauGlobComb['Variable source'] == i,'rang 3'] = 1
        TableauGlobComb.loc[TableauGlobComb['Variable source'] == i,'rang 1'] = 1
    else:
        if(Valeur in df_eval3["var. cand. avec les meilleurs scores sur la combinaison"].values):
            TableauGlobComb.loc[TableauGlobComb['Variable source'] == i,'rang 3'] = 1
            TableauGlobComb.loc[TableauGlobComb['Variable source'] == i,'rang 5'] = 1
            TableauGlobComb.loc[TableauGlobComb['Variable source'] == i,'rang 10'] = 1
        else:
            if(Valeur in df_eval5["var. cand. avec les meilleurs scores sur la combinaison"].values):
                TableauGlobComb.loc[TableauGlobComb['Variable source'] == i,'rang 5'] = 1
                TableauGlobComb.loc[TableauGlobComb['Variable source'] == i,'rang 10'] = 1
            else:
                if(Valeur in df_eval10["var. cand. avec les meilleurs scores sur la combinaison"].values): 
                    TableauGlobComb.loc[TableauGlobComb['Variable source'] == i,'rang 10'] = 1
    num_var_src = num_var_src + 1


# In[16]:



###
columns = [ ["Variable source","cosinus(nom de variables)","cosinus(nom de variables)","cosinus(nom de variables)","cosinus(nom de variables)","cosinus(description de variables)","cosinus(description de variables)","cosinus(description de variables)","cosinus(description de variables)","Combinaison","Combinaison","Combinaison","Combinaison"],
                ["","précision au rang 1","Rang 3","Rang 5","Rang 10","précision au rang 1","Rang 3","Rang 5","Rang 10","précision au rang 1","Rang 3","Rang 5","Rang 10"]
                ]

tuples = list(zip(*columns))

index = pd.MultiIndex.from_tuples(tuples)
Tableau_avec_lemma = pd.DataFrame( columns=index)

    
# tableau d'évaluation de Levenshtein et du cosinus
#Tableau_avec_lemma = pd.DataFrame( columns=index)

Tableau_avec_lemma["Variable source"] = varSrcList
Tableau_avec_lemma["cosinus(nom de variables)","précision au rang 1"] = TableauGlobLev["rang 1"]
Tableau_avec_lemma["cosinus(nom de variables)","Rang 3"] = TableauGlobLev["rang 3"]
Tableau_avec_lemma["cosinus(nom de variables)","Rang 5"] = TableauGlobLev["rang 5"]
Tableau_avec_lemma["cosinus(nom de variables)","Rang 10"] = TableauGlobLev["rang 10"]
Tableau_avec_lemma["cosinus(description de variables)","précision au rang 1"] = TableauGlobCos["rang 1"]
Tableau_avec_lemma["cosinus(description de variables)","Rang 3"] = TableauGlobCos["rang 3"]
Tableau_avec_lemma["cosinus(description de variables)","Rang 5"] = TableauGlobCos["rang 5"]
Tableau_avec_lemma["cosinus(description de variables)","Rang 10"] = TableauGlobCos["rang 10"]
Tableau_avec_lemma["Combinaison","précision au rang 1"] = TableauGlobComb["rang 1"]
Tableau_avec_lemma["Combinaison","Rang 3"] = TableauGlobComb["rang 3"]
Tableau_avec_lemma["Combinaison","Rang 5"] = TableauGlobComb["rang 5"]
Tableau_avec_lemma["Combinaison","Rang 10"] = TableauGlobComb["rang 10"]

Tableau_avec_lemma.to_csv(f'C:\\Users\\PC\\Desktop\\rapport stage M2\\Stage M2\\BERT-base et TF-IDF\\Evaluation.csv')

html = Tableau_avec_lemma.to_html()
html_TG_Lev_cos = open(f'C:\\Users\\PC\\Desktop\\rapport stage M2\\Stage M2\\BERT-base et TF-IDF\\Evaluation.html', "w")
html_TG_Lev_cos.write(html)
html_TG_Lev_cos.close()


# In[17]:


nbre_varSrc = len(varSrcList)

############ Récapitatif des résultats
    
###
columns = [ ["Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","Nombre de bons résultats après lemmatisation (sur les 84 variables sources)"],
                ["cosinus(nom de variables)","cosinus(nom de variables)","cosinus(nom de variables)","cosinus(nom de variables)","cosinus(description de variables)","cosinus(description de variables)","cosinus(description de variables)","cosinus(description de variables)","Combinaison","Combinaison","Combinaison","Combinaison"],
                ["précision au rang 1","Rang 3","Rang 5","Rang 10","précision au rang 1","Rang 3","Rang 5","Rang 10","précision au rang 1","Rang 3","Rang 5","Rang 10"]
                ]

tuples = list(zip(*columns))

index = pd.MultiIndex.from_tuples(tuples)
# tableau d'évaluation de Levenshtein et du cosinus
Recap_avec_lemma = pd.DataFrame( columns=index)


# In[18]:



    #Levenshtein

resultat = np.sum(Tableau_avec_lemma["cosinus(nom de variables)","précision au rang 1"])/nbre_varSrc
Recap_avec_lemma["Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","cosinus(nom de variables)","précision au rang 1"] = ["{:.2%}".format(resultat)]

resultat = np.sum(Tableau_avec_lemma["cosinus(nom de variables)","Rang 3"])/nbre_varSrc
Recap_avec_lemma["Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","cosinus(nom de variables)","Rang 3"] = ["{:.2%}".format(resultat)]

resultat = np.sum(Tableau_avec_lemma["cosinus(nom de variables)","Rang 5"])/nbre_varSrc
Recap_avec_lemma["Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","cosinus(nom de variables)","Rang 5"] = ["{:.2%}".format(resultat)]

resultat = np.sum(Tableau_avec_lemma["cosinus(nom de variables)","Rang 10"])/nbre_varSrc
Recap_avec_lemma["Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","cosinus(nom de variables)","Rang 10"] = ["{:.2%}".format(resultat)]


# In[19]:



#Cosinus

resultat = np.sum(Tableau_avec_lemma["cosinus(description de variables)","précision au rang 1"])/nbre_varSrc
Recap_avec_lemma["Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","cosinus(description de variables)","précision au rang 1"] = ["{:.2%}".format(resultat)]

resultat = np.sum(Tableau_avec_lemma["cosinus(description de variables)","Rang 3"])/nbre_varSrc
Recap_avec_lemma["Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","cosinus(description de variables)","Rang 3"] = ["{:.2%}".format(resultat)]

resultat = np.sum(Tableau_avec_lemma["cosinus(description de variables)","Rang 5"])/nbre_varSrc
Recap_avec_lemma["Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","cosinus(description de variables)","Rang 5"] = ["{:.2%}".format(resultat)]

resultat = np.sum(Tableau_avec_lemma["cosinus(description de variables)","Rang 10"])/nbre_varSrc
Recap_avec_lemma["Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","cosinus(description de variables)","Rang 10"] = ["{:.2%}".format(resultat)]


# In[20]:



#Combinaison

resultat = np.sum(Tableau_avec_lemma["Combinaison","précision au rang 1"])/nbre_varSrc
Recap_avec_lemma["Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","Combinaison","précision au rang 1"] = ["{:.2%}".format(resultat)]

resultat = np.sum(Tableau_avec_lemma["Combinaison","Rang 3"])/nbre_varSrc
Recap_avec_lemma["Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","Combinaison","Rang 3"] = ["{:.2%}".format(resultat)]

resultat = np.sum(Tableau_avec_lemma["Combinaison","Rang 5"])/nbre_varSrc
Recap_avec_lemma["Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","Combinaison","Rang 5"] = ["{:.2%}".format(resultat)]

resultat = np.sum(Tableau_avec_lemma["Combinaison","Rang 10"])/nbre_varSrc
    
Recap_avec_lemma["Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","Combinaison","Rang 10"] = ["{:.2%}".format(resultat)]


# In[21]:



html = Recap_avec_lemma.to_html()
html_TG_Lev_cos = open(f'C:\\Users\\PC\\Desktop\\rapport stage M2\\Stage M2\\BERT-base et TF-IDF\\Recapitulatif.html', "w")
html_TG_Lev_cos.write(html)
html_TG_Lev_cos.close()


# In[62]:


#################################################################################### 0.25

