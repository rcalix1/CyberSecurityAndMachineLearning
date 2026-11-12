## LLM text Water Marking

* A Watermark for Large Language Models
* John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, Tom Goldstein
* Proceedings of the 40th International Conference on Machine Learning, PMLR 202:17061-17084, 2023.
* https://proceedings.mlr.press/v202/kirchenbauer23a.html

---

## Code




```python


import numpy as np

np.random.seed(42)

# Vocabulary and original probabilities
words = ["cat", "dog", "bird", "fish", "horse", "mouse"]
probs = np.array([0.30, 0.25, 0.20, 0.10, 0.10, 0.05])

# Secret green list (normally generated using a secret key)
green = {"cat", "bird", "horse"}

# Watermark: boost probabilities of green words
wm_probs = np.array([
    p * 2 if w in green else p
    for w, p in zip(words, probs)
])
wm_probs /= wm_probs.sum()

# Generate 100 tokens
normal = np.random.choice(words, 100, p=probs)
watermarked = np.random.choice(words, 100, p=wm_probs)

# Count green tokens
def detect(tokens):
    count = sum(w in green for w in tokens)
    return count, count / len(tokens)

print("Original probabilities:", probs)
print("Watermarked probabilities:", wm_probs)

print("\nNormal:", detect(normal))
print("Watermarked:", detect(watermarked))


```





