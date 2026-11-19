## Cryptanalysis and ML



# Machine Learning and Cryptanalysis

## 1. Can Machine Learning Break Encryption?

Cryptanalysis is the study of techniques used to analyze and break cryptographic systems. Traditionally, cryptanalysis relies on mathematical properties, statistical analysis, or exhaustive key searches.

An interesting research question is whether **machine learning can learn encryption and decryption operations directly from examples**.

In this experiment, we investigate whether a neural network can learn to decrypt a simple Caesar cipher.

### The Caesar Cipher

The Caesar cipher is a substitution cipher in which each letter is shifted by a fixed number of positions in the alphabet.

For example, with a key of 3:

- A becomes D
- B becomes E
- C becomes F
- Z becomes C

Each letter is represented by an integer between 0 and 25.

Encryption is defined as:

$$
c = (p+k) \bmod 26
$$

Decryption is defined as:

$$
p = (c-k) \bmod 26
$$

where:

- $p$ is the plaintext letter.
- $c$ is the ciphertext letter.
- $k$ is the encryption key.

## 2. Neural Network Cryptanalysis

Instead of using the mathematical decryption formula, we train a neural network to learn the relationship between ciphertext and plaintext.

The model learns a function:

$$
\hat{p}=f_\theta(c)
$$

where $\theta$ represents the neural network parameters.

The objective is to minimize the classification loss:

$$
L(\theta)=-\sum_{i=1}^{N}\log P_\theta(p_i\mid c_i)
$$

This is the cross-entropy loss for predicting the correct plaintext letter.

### Python Experiment 1: Learning to Decrypt

We generate 1,000 random plaintext letters and encrypt them using a fixed Caesar cipher key.

The ciphertext letters are converted into one-hot vectors with 26 elements.

We train a neural network with:

- 26 input features
- 64 hidden neurons
- 26 output classes
- 800 training samples
- 200 testing samples

The neural network attempts to predict the original plaintext letter from its encrypted representation.

### Experiments

1. Change the encryption key from 3 to 7 and retrain the model.
2. Reduce the number of training samples from 800 to 20.
3. Increase or decrease the number of hidden neurons.
4. Test whether the model can decrypt letters that were absent from its training data.

### Discussion

The neural network can learn the Caesar cipher because the encryption transformation is simple and deterministic.

However, there are only 26 possible ciphertext letters for a fixed key. The network can therefore memorize the mapping between ciphertext and plaintext without discovering the mathematical encryption rule.

This raises an important research question:

**Can a neural network learn a cryptographic algorithm rather than simply memorize examples?**

Modern cryptographic algorithms are designed to make this substantially more difficult.

## Reference

- Greydanus, S. (2017). *Learning the Enigma with Recurrent Neural Networks*.



```


import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Generate training data
np.random.seed(42)

key = 3

plaintext = np.random.randint(0, 26, 1000)
ciphertext = (plaintext + key) % 26

# One-hot encode the ciphertext
X = np.eye(26)[ciphertext]
y = plaintext

# Train on 800 samples
model = MLPClassifier(
    hidden_layer_sizes=(64,),
    max_iter=1000,
    random_state=42
)

model.fit(X[:800], y[:800])

# Test on 200 samples
pred = model.predict(X[800:])

print("Accuracy:", accuracy_score(y[800:], pred))

# Try decrypting a letter
letter = 10  # K
encrypted = (letter + key) % 26

prediction = model.predict(
    np.eye(26)[[encrypted]]
)[0]

print("Original letter:", chr(letter + 65))
print("Encrypted letter:", chr(encrypted + 65))
print("ML decrypted letter:", chr(prediction + 65))




```



















