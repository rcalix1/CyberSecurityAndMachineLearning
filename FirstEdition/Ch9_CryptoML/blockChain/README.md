## Blockchain

* Proof of work




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

































