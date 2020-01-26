import pandas as pd
from sklearn.externals import joblib
import re, nltk, spacy, gensim
from bs4 import BeautifulSoup
from nltk.tokenize import ToktokTokenizer
from nltk.stem import wordnet
from nltk.corpus import stopwords
import en_core_web_sm
from string import punctuation
from sklearn.externals import joblib
#nltk.download()
nltk.download('stopwords')

def formulaire(request):
    tags = None
    
    if request.method == 'POST':
        
        question = request.form["question"].upper()
        tags = predict(question)	
	 
    return tags	 

	
def predict(question):
    # faire ce qu'il faut pour calculer le retard
    text=question
    top_tags=pd.read_csv("./top_tags.csv")
        
    # Nettoyage texte
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"\'\n", " ", text)
        text = re.sub(r"\'\xa0", " ", text)
        text = re.sub('\s+', ' ', text) # matches all whitespace characters
        text = text.strip(' ')
        return text

    token = ToktokTokenizer()
    punct = punctuation

    # Suppression nul & strip
    def strip_list(mylist):
        newlist = (item.strip() if hasattr(item, 'strip') else item for item in mylist)
        return [item for item in newlist if item != '']

    # Suppression ponctuation
    def ponctuation(text): 
        words = token.tokenize(text)
        punctuation_filtered = []
        regex = re.compile('[%s]' % re.escape(punct))
        remove_punctuation = str.maketrans(' ', ' ', punct)

        for w in words:
            if w in top_tags:
                punctuation_filtered.append(w)
            else:
                w = re.sub('^[0-9]*', " ", w)
                punctuation_filtered.append(regex.sub('', w))

        filtered_list = strip_list(punctuation_filtered)

        return ' '.join(map(str, filtered_list)) 

    stop_words = set(stopwords.words("english"))

    # Suppression stopwords
    def stopWords(text):
        words = token.tokenize(text)
        filtered = [w for w in words if not w in stop_words]
        return ' '.join(map(str, filtered))

    nlp = en_core_web_sm.load()

    # Lemmatisation 
    def lemmatization(texts, allowed_postags, stop_words=stop_words):
        lemma = wordnet.WordNetLemmatizer()       
        doc = nlp(texts) 
        texts_out = []
        for token in doc:
            if str(token) in top_tags.values:
                texts_out.append(str(token))
            elif token.pos_ in allowed_postags:
                if token.lemma_ not in ['-PRON-']:
                    texts_out.append(token.lemma_)
                else:
                    texts_out.append('')
        texts_out = ' '.join(texts_out)

        return texts_out
                
    model = joblib.load("./models/LR.joblib")
    Tfidf = joblib.load("./models/Tfidf.joblib")
    label = joblib.load("./models/label.joblib")
                
    text = clean_text(text)
    text = ponctuation(text)
    text = stopWords(text)
    text = lemmatization(text, ['NOUN', 'ADV'])
    list_words = []


    text_tfidf = Tfidf.transform([text])

    y_pred=model.predict(text_tfidf)

    for i in range(0,99):
        if y_pred[0][i] == 1:
            list_words.append(label.classes_[i])
                        
        tags = " ".join(list_words)
                
    return tags
    

