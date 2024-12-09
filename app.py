from flask import Flask, request, render_template
import joblib
from feature import get_all_query, remove_punctuation_stopwords_lemma

app = Flask(__name__)

# Load the model pipeline
model = joblib.load('pipeline.sav')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    title = request.form['title']
    author = request.form['author']
    text = request.form['text']
    
    # Preprocess the input
    query = get_all_query(title, author, text)
    processed_query = remove_punctuation_stopwords_lemma(query)
    
    # Make prediction
    prediction = model.predict([processed_query])[0]
    
    # Render result
    return render_template('index.html', prediction_text=f'This news is likely to be {prediction}')

if __name__ == '__main__':
    app.run(debug=True)
