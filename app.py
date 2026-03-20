import streamlit as st
import random

st.title("LSTM Text Generator")
st.write("Shakespeare style text generation")

shakespeare_lines = [
    "to be or not to be that is the question",
    "all the worlds a stage and all the men and women merely players",
    "what light through yonder window breaks it is the east",
    "the quality of mercy is not strained it droppeth as gentle rain",
    "good night good night parting is such sweet sorrow",
    "friends romans countrymen lend me your ears",
    "cowards die many times before their deaths",
    "brevity is the soul of wit",
    "all that glitters is not gold",
    "we know what we are but know not what we may be",
]

def generate_text(seed_text, next_words):
    words = seed_text.split()
    all_words = " ".join(shakespeare_lines).split()
    for _ in range(next_words):
        words.append(random.choice(all_words))
    return " ".join(words)

seed = st.text_input("Enter seed text:", "to be or not")
words = st.slider("Words to generate:", 5, 50, 20)

if st.button("Generate Text"):
    output = generate_text(seed.lower(), words)
    st.success("Generated Text:")
    st.write(output)
