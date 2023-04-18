import numpy as np
import pandas as pd
import scipy

#nltk-preprocessing
import string
import nltk
import contractions
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.stem.wordnet import WordNetLemmatizer


#misc
import re
import pickle
import joblib
import warnings
warnings.filterwarnings("ignore")
from collections.abc import Iterable

#metrics
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve, auc,roc_auc_score

#model loading
from tensorflow.keras.models import load_model
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')

def convert_to_lower_case(text):
    
    """function to convert the input text to lower case"""
    
    return text.lower()

def remove_escape_char(text):
    
    """function to remove newline (\n),
    tab(\t) and slashes (/ , \) from the input text"""

    return re.sub(r"[\n\t\\\/]"," ",text, flags=re.MULTILINE)

def remove_html_tags(text):
    
    """function to remove html tags (< >) and its content 
    from the input text"""

    return re.sub(r"<.*>"," ",text, flags=re.MULTILINE)

def remove_links(text):
    """function to remove any kind of links with no 
    html tags"""

    text= re.sub(r"http\S+"," ",text, flags=re.MULTILINE)

    return re.sub(r"www\S+"," ",text, flags=re.MULTILINE)

def remove_digits(text):
    
    """function to remove digits from the input text"""

    return re.sub(r'\d'," ",text, flags=re.MULTILINE)

def remove_punctuation(text):
    
    """function to remove punctuation marks from the input text"""

    for i in string.punctuation:
        text = text.replace(i," ")

    return text      

def chuncking(text):
    
    """function to perform chucking, which is also referred as shallow parsing.
    This is useful in determing the parts of speech of a given text and adds more
    structure to the input data ."""

    """In this function, we use NLTK library to perform chuncking and if a 
    particular label is PERSON names, we remove that, and names of Geo-graphic
    ares are retained by adding _ in its words.ex-New_York"""


    chunks_data=[]
    chunks_data=(list(ne_chunk(pos_tag(word_tokenize(text)))))
    for label in chunks_data:
        if type(label)==Tree:
            if label.label() == "GPE":
                a = label.leaves()
                if len(a)>1:
                    gpe = "_".join([term for term,pos in a])
                    text = re.sub(rf'{a[1][0]}',gpe,text, flags=re.MULTILINE)
                    text = re.sub(rf'\b{a[0][0]}\b'," ",text, flags=re.MULTILINE)
            if label.label()=="PERSON":      
                for term,pog in label.leaves():
                    text = re.sub(re.escape(term)," ",text, flags=re.MULTILINE)
    return text

def keep_alpha_and_underscore(text):
    
    """function to keep only aphabets and _ underscore, as we 
    added it in the chunking for geographic locations."""
    
    return re.sub(r"[^a-zA-Z_]"," ",text,flags=re.MULTILINE)

def remove_extra_spaces_if_any(text):
    
    """function to remove extra spaces if any after all the pre-preocessing"""
    
    return re.sub(r" {2,}", " ", text, flags=re.MULTILINE)

def remove_repeated_characters(text):
    
    """function to remove repeated characters if any from the input text"""

    """for example CAAAAASSSSSSEEEEE SSSSTTTTTUUUUUUDDDDYYYYYY gives CASE STUDY"""

    return re.sub(r"(\w)(\1{2,})","\\1",text,flags=re.MULTILINE)

def remove_words_lesth2(text):
    """function to remove words with length less than 2"""

    text = re.sub(r'\b\w{1,2}\b'," ",text)
    
    return text

def decontraction(text):
    
    """function to handle contraction errors"""
    res=""
    for word in text.split():
        try:
            con_text=contractions.fix(word)
            if con_text.lower() is word.lower():
                res=res+word+" "
            else:
                res=res+con_text+" "
        
        except:
            con_text=contractions.fix(word.lower())
            if con_text.lower() is word.lower():
                res=res+word+" "
            else:
                res=res+con_text+" "
    return res.strip()

#lets take all the stop words from both NLTK & Word Cloud libraries, along 
# with some custom words

stop_words=stopwords.words('english')
word_cloud_stp_wrds=list(STOPWORDS)
final_stop_words=list(STOPWORDS.union(set(stop_words)))
final_stop_words.extend(["mr","mrs","miss",
                        "one","two","three","four","five",
                        "six","seven","eight","nine","ten",
                        "us","also","dont","cant","any","can","along",
                        "among","during","anyone",
                         "a","b","c","d","e","f","g","h","i","j","k","l","m",
                         "n","o","p","q","r","s","t","u","v","w","x","y","z","hi","hello","hey","ok",
                         "okay","lol","rofl","hola","let","may","etc"])

#lemmatizer object
lemmatiser = WordNetLemmatizer()

# one-step pre-processing function

def preprocess(text):

    preprocessed_text = []

    for each_text in text:

        result=remove_links(each_text)
        result=remove_html_tags(result)
        result=remove_escape_char(result)        
        result=remove_digits(result)
        result=decontraction(result)
        result=remove_punctuation(result)
        result=chuncking(result)
        result=convert_to_lower_case(result)
        result = ' '.join(non_stop_word for non_stop_word in result.split() if non_stop_word not in final_stop_words)
        result=keep_alpha_and_underscore(result)
        result=remove_extra_spaces_if_any(result)
        result=remove_repeated_characters(result)
        result=remove_words_lesth2(result)
        result=' '.join(lemmatiser.lemmatize(word,pos="v") for word in result.split())
        preprocessed_text.append(result.strip())
        
    return preprocessed_text

#load data

tfidf_dict = joblib.load(r'C:\Users\nimis_r\OneDrive\Desktop\INTERNSHIPS\Temp\twitter-classifier-backend\ml_models\tfidf_dict.pkl')
tfidf_words = joblib.load(r'C:\Users\nimis_r\OneDrive\Desktop\INTERNSHIPS\Temp\twitter-classifier-backend\ml_models\tfidf_words.pkl')
w2v_dict = joblib.load(r'C:\Users\nimis_r\OneDrive\Desktop\INTERNSHIPS\Temp\twitter-classifier-backend\ml_models\w2v_dict.pkl')
w2v_words = joblib.load(r'C:\Users\nimis_r\OneDrive\Desktop\INTERNSHIPS\Temp\twitter-classifier-backend\ml_models\w2v_words.pkl')

# computing tf-idf weighted word2vec for each comment.

def comp_tfidf_weighted_w2v(data,w2v_words,tfidf_words,w2v_dict,tfidf_dict):

    tfidf_w2v = []
    for sentence in data:
        vector = np.zeros(300) 
        # as word vectors are of zero length
        tf_idf_weight =0;
        # num of words with a valid vector in the sentence/review
        for word in sentence.split(): 
            # for each word in a review/sentence
            if (word in w2v_words) and (word in tfidf_words):
                vec = w2v_dict[word] 
                # getting the vector for each word
                # here we are multiplying idf value(dictionary[word]) and 
                #the tf value((sentence.count(word)/len(sentence.split())))
                tf_idf = tfidf_dict[word]*(sentence.count(word)/len(sentence.split()))
                # getting the tfidf value for each word
                vector += (vec * tf_idf) # calculating tfidf weighted w2v
                tf_idf_weight += tf_idf
        if tf_idf_weight != 0:
            vector /= tf_idf_weight
        tfidf_w2v.append(vector)
    return np.array(tfidf_w2v)

#loading model
model=load_model(r"C:\Users\nimis_r\OneDrive\Desktop\INTERNSHIPS\Temp\twitter-classifier-backend\ml_models\nlp_model.h5", compile=False)
model.save(r"C:\Users\nimis_r\OneDrive\Desktop\INTERNSHIPS\Temp\twitter-classifier-backend\ml_models\best_model.hdf5")

def cal_metrics(y_true,y_pred):
    
    """function to calculate final metrics """

    if isinstance(y_true,scipy.sparse.lil.lil_matrix):
        y_true=y_true.A
    
    if isinstance(y_pred,scipy.sparse.lil.lil_matrix):
        y_pred=y_pred.A

    acc=accuracy_score(y_true,y_pred)
    ham_loss=hamming_loss(y_true,y_pred)

    return {"Accuracy":acc,"Hamming Loss":ham_loss}

def function_1(X):
    
    #handling single & multiple inputs

    if isinstance(X,str):
        X=[X]

    elif isinstance(X,Iterable):
        X=X

    #pre-processing
    pp_text=preprocess(X)

    #vectorizing
    vect_data=comp_tfidf_weighted_w2v(pp_text,w2v_words,
                                      tfidf_words,
                                      w2v_dict,
                                      tfidf_dict)
    pred=model.predict(vect_data).round().astype(int)
    
    return pred
