import streamlit as st
import numpy as np
import pickle
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.title("LSTM Text Generator")
st.write("Shakespeare style text generation")

@st.cache_resource
def load_model():
    model = keras.models.load_model('lstm_model.h5', compile=False)
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('maxlen.pkl', 'rb') as f:
        max_len = pickle.load(f)
    return model, tokenizer, max_len

model, tokenizer, max_len = load_model()

def generate_text(seed_text, next_words):
    result = seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([result])[0]
        token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                result += " " + word
                break
    return result

seed = st.text_input("Enter seed text:", "to be or not")
words = st.slider("Words to generate:", 5, 50, 20)

if st.button("Generate Text"):
    with st.spinner("Generating..."):
        output = generate_text(seed.lower(), words)
        st.success("Generated Text:")
        st.write(output)
