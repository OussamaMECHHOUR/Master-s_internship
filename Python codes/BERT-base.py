#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


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
    
with open('D:\\WISD-S3\\mon stage pfe\\mon stage pfe\\code Billy Ngaba\\projet\\plantedeservices-master\\Code\\mes executions\\fichier text_mining_complement_approche_contextuelle\\complement_approche_contextuelle.txt', "r",encoding='utf-8') as file:
    lines_contexte = file.read().splitlines()
    file.close()


# In[3]:


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


# In[4]:


#################################################### BERT-base ##################################################################


# In[5]:


import torch
from transformers import BertTokenizer, BertModel, BertConfig

# Vérifier si un GPU est disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = BertConfig(num_hidden_layers=2)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', config=config)
model = BertModel.from_pretrained('bert-base-uncased', config=config).to(device)

src_inputs = tokenizer(lines_des_src_pre5, padding=True, truncation=True, return_tensors="pt").to(device)
src_outputs = model(**src_inputs)

cand_inputs = tokenizer(lines_des_cand_pre5, padding=True, truncation=True, return_tensors="pt").to(device)
cand_outputs = model(**cand_inputs)

similarity_scores = torch.cosine_similarity(src_outputs.last_hidden_state.mean(dim=1).unsqueeze(1),
                                            cand_outputs.last_hidden_state.mean(dim=1).unsqueeze(0), dim=2)



src_inputs_leven = tokenizer(lines_var_src_pre3, padding=True, truncation=True, return_tensors="pt").to(device)
src_outputs_leven = model(**src_inputs_leven)

cand_inputs_leven = tokenizer(lines_var_cand_pre3, padding=True, truncation=True, return_tensors="pt").to(device)
cand_outputs_leven = model(**cand_inputs_leven)

similarity_scores_leven = torch.cosine_similarity(src_outputs_leven.last_hidden_state.mean(dim=1).unsqueeze(1),
                                                  cand_outputs_leven.last_hidden_state.mean(dim=1).unsqueeze(0), dim=2)


# In[6]:


##########################################################  fin BERT-base ######################################################


# In[7]:


## Evaluation
def evaluation(numero,x, rang,lines_var_src, lines_var_cand ):
    i = numero
    cosinus1 = [] # avec les des. cand
    lev = []
    combi = []
    # x dans [0,1]
    for k in range(len(lines_var_cand)):
        cosinus1.append(similarity_scores[i][k].item())
        lev.append(similarity_scores_leven[i][k].item())
        combi.append(float(x)*similarity_scores[i][k].item() + (1-float(x))*similarity_scores_leven[i][k].item())
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


# In[8]:


####################### Tableaux des évaluations

correspondances = pd.read_excel("D:\\WISD-S3\\mon stage pfe\\mon stage pfe\\code Billy Ngaba\\STAGE_M2_SSD\\Correspondances.xlsx")
varSrcList = correspondances["Variable source"]


# In[9]:


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
    df_eval10 = evaluation(num_var_src, 0.79, 10, lines_var_src1, lines_var_cand1)
    df_eval5 = evaluation(num_var_src, 0.79, 5, lines_var_src1, lines_var_cand1)
    df_eval3 = evaluation(num_var_src, 0.79, 3, lines_var_src1, lines_var_cand1)
    df_eval1 = evaluation(num_var_src, 0.79, 1, lines_var_src1, lines_var_cand1)
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
    


# In[10]:


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
    df_eval10 = evaluation(num_var_src, 0.79, 10, lines_var_src1, lines_var_cand1)
    df_eval5 = evaluation(num_var_src, 0.79, 5, lines_var_src1, lines_var_cand1)
    df_eval3 = evaluation(num_var_src, 0.79, 3, lines_var_src1, lines_var_cand1)
    df_eval1 = evaluation(num_var_src, 0.79, 1, lines_var_src1, lines_var_cand1)
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


# In[11]:


TableauGlobComb = pd.DataFrame({'Variable source':varSrcList,'rang 1': [0]*len(lines_var_src),'rang 3': [0]*len(lines_var_src),'rang 5': [0]*len(lines_var_src),'rang 10': [0]*len(lines_var_src) })


###### combi 
    
num_var_src = 0
for i in varSrcList: 
    df_eval10 = evaluation(num_var_src,0.79, 10,lines_var_src1, lines_var_cand1 )
    df_eval5 = evaluation(num_var_src,0.79, 5,lines_var_src1, lines_var_cand1 )
    df_eval3 = evaluation(num_var_src,0.79,3,lines_var_src1, lines_var_cand1 )
    df_eval1 = evaluation(num_var_src,0.79, 1,lines_var_src1, lines_var_cand1 )
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


# In[12]:



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

Tableau_avec_lemma.to_csv(f'C:\\Users\\PC\\Desktop\\rapport stage M2\\Stage M2\\BERT-base\\Evaluation.csv')

html = Tableau_avec_lemma.to_html()
html_TG_Lev_cos = open(f'C:\\Users\\PC\\Desktop\\rapport stage M2\\Stage M2\\BERT-base\\Evaluation.html', "w")
html_TG_Lev_cos.write(html)
html_TG_Lev_cos.close()


# In[13]:


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


# In[14]:



    #Levenshtein

resultat = np.sum(Tableau_avec_lemma["cosinus(nom de variables)","précision au rang 1"])/nbre_varSrc
Recap_avec_lemma["Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","cosinus(nom de variables)","précision au rang 1"] = ["{:.2%}".format(resultat)]

resultat = np.sum(Tableau_avec_lemma["cosinus(nom de variables)","Rang 3"])/nbre_varSrc
Recap_avec_lemma["Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","cosinus(nom de variables)","Rang 3"] = ["{:.2%}".format(resultat)]

resultat = np.sum(Tableau_avec_lemma["cosinus(nom de variables)","Rang 5"])/nbre_varSrc
Recap_avec_lemma["Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","cosinus(nom de variables)","Rang 5"] = ["{:.2%}".format(resultat)]

resultat = np.sum(Tableau_avec_lemma["cosinus(nom de variables)","Rang 10"])/nbre_varSrc
Recap_avec_lemma["Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","cosinus(nom de variables)","Rang 10"] = ["{:.2%}".format(resultat)]


# In[15]:



#Cosinus

resultat = np.sum(Tableau_avec_lemma["cosinus(description de variables)","précision au rang 1"])/nbre_varSrc
Recap_avec_lemma["Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","cosinus(description de variables)","précision au rang 1"] = ["{:.2%}".format(resultat)]

resultat = np.sum(Tableau_avec_lemma["cosinus(description de variables)","Rang 3"])/nbre_varSrc
Recap_avec_lemma["Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","cosinus(description de variables)","Rang 3"] = ["{:.2%}".format(resultat)]

resultat = np.sum(Tableau_avec_lemma["cosinus(description de variables)","Rang 5"])/nbre_varSrc
Recap_avec_lemma["Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","cosinus(description de variables)","Rang 5"] = ["{:.2%}".format(resultat)]

resultat = np.sum(Tableau_avec_lemma["cosinus(description de variables)","Rang 10"])/nbre_varSrc
Recap_avec_lemma["Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","cosinus(description de variables)","Rang 10"] = ["{:.2%}".format(resultat)]


# In[16]:



#Combinaison

resultat = np.sum(Tableau_avec_lemma["Combinaison","précision au rang 1"])/nbre_varSrc
Recap_avec_lemma["Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","Combinaison","précision au rang 1"] = ["{:.2%}".format(resultat)]

resultat = np.sum(Tableau_avec_lemma["Combinaison","Rang 3"])/nbre_varSrc
Recap_avec_lemma["Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","Combinaison","Rang 3"] = ["{:.2%}".format(resultat)]

resultat = np.sum(Tableau_avec_lemma["Combinaison","Rang 5"])/nbre_varSrc
Recap_avec_lemma["Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","Combinaison","Rang 5"] = ["{:.2%}".format(resultat)]

resultat = np.sum(Tableau_avec_lemma["Combinaison","Rang 10"])/nbre_varSrc
    
Recap_avec_lemma["Nombre de bons résultats après lemmatisation (sur les 84 variables sources)","Combinaison","Rang 10"] = ["{:.2%}".format(resultat)]


# In[17]:



html = Recap_avec_lemma.to_html()
html_TG_Lev_cos = open(f'C:\\Users\\PC\\Desktop\\rapport stage M2\\Stage M2\\BERT-base\\Recapitulatif.html', "w")
html_TG_Lev_cos.write(html)
html_TG_Lev_cos.close()


# In[32]:


#################################################################################### 0.79

