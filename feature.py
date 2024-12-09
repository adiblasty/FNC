import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def get_all_query(title, author, text):
    return title + author + text

def remove_punctuation_stopwords_lemma(sentence):
    sentence = re.sub(r'[^\w\s]', '', sentence)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = sentence.split()
    words = [word for word in words if word.lower() not in stop_words]
    words = [lemmatizer.lemmatize(word).lower() for word in words]
    return ' '.join(words)
