import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.title('LSTM Text Generator')

seed_text = st.text_input('Seed text', value='romeo and juliet')
num_words = st.slider('Number of words to generate', min_value=5, max_value=50, value=20)

@st.cache_resource
def load_resources():
    model = load_model('lstm_model.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('maxlen.pkl', 'rb') as f:
        maxlen = pickle.load(f)
    return model, tokenizer, maxlen

model, tokenizer, maxlen = load_resources()

if st.button('Generate'):
    text = seed_text.lower()
    output = text

    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=maxlen-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = predicted.argmax(axis=-1)[0]

        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                predicted_word = word
                break
        else:
            predicted_word = ''

        output += ' ' + predicted_word
        text += ' ' + predicted_word

    st.subheader('Generated Text')
    st.write(output)
