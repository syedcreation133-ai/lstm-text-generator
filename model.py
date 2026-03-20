import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical

# small Shakespeare dataset inside the code
text = """
Romeo and Juliet
Two households, both alike in dignity,
In fair Verona, where we lay our scene,
From ancient grudge break to new mutiny,
Where civil blood makes civil hands unclean.
From forth the fatal loins of these two foes
A pair of star-cross'd lovers take their life;
Whose misadventured piteous overthrows
Doth with their death bury their parents' strife.
The fearful passage of their death-mark'd love,
And the continuance of their parents' rage,
Which, but their children's end, nought could remove,
Is now the two hours' traffic of our stage;
The which if you with patient ears attend,
What here shall miss, our toil shall strive to mend.
""".lower()

# prepare tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
word_index = tokenizer.word_index

# create sequences
tokens = tokenizer.texts_to_sequences([text])[0]
input_sequences = []
for i in range(2, len(tokens) + 1):
    input_sequences.append(tokens[:i])

maxlen = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=maxlen, padding='pre')

X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = to_categorical(y, num_classes=len(word_index) + 1)

vocab_size = len(word_index) + 1

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=maxlen - 1))
model.add(LSTM(100, return_sequences=False))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Starting training for 100 epochs...')
model.fit(X, y, epochs=100, batch_size=32, verbose=2)

model.save('lstm_model.h5')
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
with open('maxlen.pkl', 'wb') as f:
    pickle.dump(maxlen, f)

print('Training complete. Saved lstm_model.h5, tokenizer.pkl, maxlen.pkl.')
