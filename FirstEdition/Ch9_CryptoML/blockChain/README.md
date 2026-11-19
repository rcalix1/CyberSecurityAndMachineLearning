## Blockchain

* Proof of work


---


# Bitcoin Proof of Work and Machine Learning

## 1. Bitcoin and Proof of Work

Bitcoin is a decentralized digital currency that uses a blockchain to maintain a record of transactions. A blockchain consists of blocks connected through cryptographic hashes.

Bitcoin uses a **Proof-of-Work (PoW)** mechanism in which miners repeatedly calculate hashes until they find one that satisfies a difficulty requirement.

A cryptographic hash function, such as SHA-256, converts an input into a fixed-length output. A small change in the input produces a substantially different and unpredictable hash.

### The Nonce

A **nonce** (number used once) is a number that miners change to generate different hashes.

In our simplified Python example, we calculate:

$$
h = SHA256(\text{message} \Vert \text{nonce})
$$

where $\Vert$ represents concatenation.

The goal is to find a nonce such that:

$$
H(\text{message} \Vert \text{nonce}) < T
$$

where $T$ is a difficulty target.

For simplicity, our program requires the hexadecimal hash to begin with a specified number of zeros.

The probability of finding a valid hash in one attempt is:

$$
P = \frac{1}{16^d}
$$

where $d$ is the number of required leading hexadecimal zeros.

For example, with difficulty 4:

$$
P = \frac{1}{16^4} = \frac{1}{65536}
$$

On average, approximately 65,536 attempts are needed.

**Experiment 1:** Change the difficulty from 2 to 5 and observe how the number of attempts and execution time increase.

## 2. Can Machine Learning Predict a Successful Nonce?

Instead of searching for a valid nonce through repeated hashing, could a machine learning algorithm predict which nonces will produce valid hashes?

In our second Python experiment, we generate 10,000 nonces and calculate their SHA-256 hashes.

We create a binary classification problem:

$$
y =
\begin{cases}
1 & \text{if the hash satisfies the difficulty}\\
0 & \text{otherwise}
\end{cases}
$$

The input feature is the nonce, and the target is whether the hash begins with `00`.

We train a Random Forest classifier using 8,000 samples and test it on 2,000 unseen samples.

### Experimental Results

Our experiment produced:

| Metric | Result |
|---|---:|
| Accuracy | 99.45% |
| Correctly predicted failures | 1,989 |
| Actual successful nonces | 11 |
| Correctly predicted successes | 0 |
| Recall for successful nonces | 0% |

Despite achieving 99.45% accuracy, the classifier failed to identify any successful nonce.

### Class Imbalance

For difficulty 2, the probability of success is:

$$
P(y=1)=\frac{1}{256}
$$

Therefore, approximately 99.61% of randomly selected nonces will fail.

A classifier that always predicts failure can achieve very high accuracy without learning anything useful.

This demonstrates why accuracy alone is often a misleading evaluation metric for highly imbalanced cybersecurity datasets.

### Why Does Machine Learning Struggle?

Cryptographic hash functions are designed to make their outputs computationally unpredictable from their inputs.

Consequently, knowing that one nonce produces a successful hash does not provide a useful pattern for predicting whether nearby nonces will succeed.

Our experiment illustrates two important ideas:

1. **Class imbalance:** High classification accuracy does not necessarily imply useful predictive performance.
2. **Cryptographic unpredictability:** Ordinary machine learning models should not be expected to discover useful predictive patterns in secure cryptographic hash outputs.

This small experiment does not prove that all ML approaches must fail, but it illustrates why brute-force hashing remains the practical approach for finding valid proof-of-work solutions.

## References

1. Nakamoto, S. (2008). *Bitcoin: A Peer-to-Peer Electronic Cash System*.
2. National Institute of Standards and Technology (2015). *Secure Hash Standard (SHS), FIPS 180-4*.
3. Breiman, L. (2001). *Random Forests*. Machine Learning, 45, 5–32.




---


```




import hashlib
import time

difficulty = 4
nonce = 0
start = time.time()

while True:
    message = f"Hello Bitcoin {nonce}"
    hash_value = hashlib.sha256(message.encode()).hexdigest()

    if hash_value.startswith("0" * difficulty):
        break

    nonce += 1

print("Nonce:", nonce)
print("Hash:", hash_value)
print("Attempts:", nonce + 1)
print("Time:", round(time.time() - start, 3), "seconds")





```



## ML to find patterns 


```


import hashlib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Generate data
X = []
y = []

for nonce in range(10000):
    message = f"Hello Bitcoin {nonce}"
    h = hashlib.sha256(message.encode()).hexdigest()

    X.append([nonce])
    y.append(int(h.startswith("00")))





import matplotlib.pyplot as plt
import numpy as np

counts = np.bincount(y, minlength=2)

print("Failures:", counts[0])
print("Successes:", counts[1])
print("Success rate:", counts[1] / len(y))

plt.bar(["Failure (0)", "Success (1)"], counts)
plt.ylabel("Number of samples")
plt.title("Class Imbalance in Bitcoin Mining")
plt.show()






X = np.array(X)
y = np.array(y)

# Train on earlier nonces, test on unseen later nonces
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X[:8000], y[:8000])

pred = model.predict(X[8000:])

print("Accuracy:", accuracy_score(y[8000:], pred))
print("Confusion matrix:")
print(confusion_matrix(y[8000:], pred, labels=[0, 1]))
print("Actual successful nonces:", sum(y[8000:]))
print("Predicted successful nonces:", sum(pred))





```

































