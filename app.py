from flask import Flask, render_template, request
import joblib
import re
import string
import spacy

# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

# Load the trained XGBoost model and label encoder
model = joblib.load('xgboost_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
# Load the TfidfVectorizer (or other transformer)
tfidfvect = joblib.load('tfidfvect.pkl')

# Define the preprocess function using spaCy tokenizer
def preprocess_text(text):
    # Remove @mentions, URLs, and hashtags
    text = re.sub(r'@+', '', text)
    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)
    text = re.sub(r'#', '', text)

    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    # Tokenize using spaCy
    tokens = [token.text for token in nlp(text)]

    # Convert tokens to lowercase and filter out non-empty tokens
    cleaned_tokens = [token.lower() for token in tokens if token.strip()]

    # Lemmatize using spaCy
    lemmatized_tokens = [token.lemma_ for token in nlp(" ".join(cleaned_tokens))]

    # Remove stop words
    cleaned_tokens = [token for token in lemmatized_tokens if token not in nlp.Defaults.stop_words]
    # Join the preprocessed tokens into a single string
    preprocessed_text = ' '.join(cleaned_tokens)

    # Return the preprocessed text as a single string
    return preprocessed_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the user's input tweet from the form
        user_tweet = request.form['tweet']

        # Preprocess the input tweet using the loaded preprocessing function (if needed)
        # Debugging print statements
        # print("User Tweet:", user_tweet)
        preprocessed_tweet = preprocess_text(user_tweet)
        # print("Preprocessed Tweet:", preprocessed_tweet)

        # Transform the preprocessed text using the loaded TfidfVectorizer
        tfidf_features = tfidfvect.transform([preprocessed_tweet])

        # Debugging print statement
        # print("TF-IDF Features:", tfidf_features)

        # Make predictions using the loaded XGBoost model
        prediction = model.predict(tfidf_features)[0]

        # Debugging print statement
        # print("Raw Prediction:", prediction)

        # Decode the predicted class label
        predicted_class = label_encoder.inverse_transform([prediction])[0]

        # Debugging print statement
        # print("Predicted Class:", predicted_class)

        # Return the predicted class to the user
        return render_template('index.html', prediction=predicted_class, tweet=user_tweet)
    except Exception as e:
        # Print the error message for debugging
        # print("Error:", str(e))
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
