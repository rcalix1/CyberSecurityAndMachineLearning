## NIO Dataset Primer

---



# GPT Data Primer Optimization Using NIO

## 1. Introduction

Can we train a neural network using a very small synthetic dataset instead of a much larger original dataset?

In this experiment, we use **Neural Inverse Optimization (NIO)** to discover a small synthetic training dataset, called a **data primer**, that allows a second GPT model to approximate the weights and behavior of a reference GPT.

This experiment also introduces a cybersecurity research question:

**Can information about a neural network's original training data be inferred from its trained weights?**

This is related to model inversion, training-data leakage, and model extraction.

## 2. Cybersecurity Connection

Consider a company that releases the weights of a trained language model, such as Meta's Llama 3.2, but does not release its complete training dataset.

An investigator has access to:

- The trained neural network weights.
- The model architecture.
- The ability to create synthetic training data.

The investigator does not have access to the original training dataset.

Can inverse optimization discover synthetic data that reproduces the model's learned behavior?

This is conceptually similar to cryptanalysis.

| Cryptanalysis | Neural Network Inversion |
|---|---|
| Ciphertext | Trained model weights |
| Unknown key | Unknown training information |
| Candidate key | Synthetic data primer |
| Search algorithm | NIO |
| Successful result | Reproducing learned behavior |

The analogy is conceptual. Neural network training is not encryption, and there is generally no unique training dataset that corresponds to a given set of weights.

## 3. Experimental Dataset

We use a simple character-level language modeling problem.

```python
text = "abc" * 200
```

This produces a dataset of 600 characters:

```text
abcabcabcabcabcabc...
```

The model learns to predict the next character.

| Input | Target |
|---|---|
| abca | bcab |
| bcab | cabc |
| cabc | abca |

The vocabulary consists of three characters:

```python
vocab = "abc"
```

This small dataset makes the experiment easy to understand and computationally inexpensive.

## 4. GPT Architecture

We use a tiny GPT-style model consisting of:

1. Character embedding layer.
2. Causal self-attention layer.
3. Linear output layer.

The embedding dimension is 8, and the model uses one attention head.

Both GPT models have **exactly the same architecture and identical initial weights**.

## 5. GPT 1 — Reference Model

GPT 1 is trained normally using the original dataset.

```text
Original Dataset (600 characters)
              |
              v
            GPT 1
              |
              v
       Trained Weights
```

The training procedure uses:

- 20 gradient-descent steps.
- Learning rate = 0.1.
- Cross-entropy loss.
- Stochastic Gradient Descent (SGD).

The resulting trained weights are saved as the reference:

\[
\theta_{\text{ref}}
\]

These are the weights NIO attempts to reproduce.

## 6. GPT 2 — Synthetic Primer Model

GPT 2 starts with the same initial weights as GPT 1.

However, GPT 2 is **never trained on the original dataset**.

Instead, it is trained on a synthetic primer containing only six positions.

Initially, the primer is random:

```python
primer = nn.Parameter(torch.randn(6, 3))
```

Each position contains three trainable values corresponding to the vocabulary `a`, `b`, and `c`.

A softmax converts these values into probabilities:

\[
P_{ij}=\frac{e^{z_{ij}}}{\sum_k e^{z_{ik}}}
\]

These continuous probabilities allow NIO to optimize the primer through backpropagation.

The primer is not initially told to contain the sequence `abc`.

## 7. Neural Inverse Optimization

The central objective is to find a synthetic primer that causes GPT 2 to learn weights similar to GPT 1.

First, GPT 2 is trained using only the primer:

\[
\theta_{\text{primer}}
=
\operatorname{Train}(\theta_0,D_{\text{primer}})
\]

Then NIO compares the resulting weights against the reference weights:

\[
\boxed{
L_{\text{NIO}}
=
\sum_l
\operatorname{Mean}
\left[
(\theta_{\text{primer}}^{(l)}
-
\theta_{\text{ref}}^{(l)})^2
\right]
}
\]

Here, \(l\) identifies a parameter tensor in the network.

NIO updates the primer to minimize this loss.

**Important:** NIO does not directly optimize GPT 2's weights. GPT 2's weights are updated through ordinary gradient-descent training on the primer.

NIO differentiates through these training steps to determine how the primer should change.

### Optimization Process

```text
             Original Dataset
                    |
                    v
                  GPT 1
                    |
                    v
             Reference Weights
                    |
                    |
Random Primer       |
      |             |
      v             |
    GPT 2           |
      |             |
      v             |
 Trained Weights    |
      |             |
      +------Compare+
               |
               v
         Weight Difference
               |
               v
         NIO Updates Primer
               |
               +---- Repeat
```

At each NIO iteration, GPT 2 starts again from the original initial weights and is trained on the current primer.

## 8. Experimental Results

The experiment used:

| Parameter | Value |
|---|---|
| Original dataset | 600 characters |
| Primer size | 6 soft-character positions |
| GPT 1 training steps | 20 |
| GPT 2 training steps | 20 |
| NIO iterations | 200 |
| GPT learning rate | 0.1 |
| NIO learning rate | 0.1 |

### NIO Weight Optimization

| NIO Iteration | Weight Difference |
|---|---:|
| 0 | 0.03518 |
| 20 | 0.00559 |
| 40 | 0.00390 |
| 60 | 0.00227 |
| 80 | 0.00138 |
| 100 | 0.00111 |
| 120 | 0.00100 |
| 140 | 0.00092 |
| 160 | 0.00087 |
| 180 | 0.00082 |

NIO reduced the weight-difference objective by approximately **97.7%** between iterations 0 and 180.

### Discovered Primer

The final primer, decoded by selecting the most probable character at each position, was:

```text
cabcac
```

The original dataset followed the repeating pattern:

```text
abcabcabc...
```

NIO was not explicitly given this pattern when initializing the primer.

Note that `cabcac` is only a readable representation. GPT 2 actually trains on the continuous probability distributions, not the decoded string.

### Final Model Comparison

Both models were evaluated on the original dataset without additional training.

| Model | Evaluation Loss |
|---|---:|
| GPT 1 — Original Dataset | 0.86926 |
| GPT 2 — NIO Primer Only | 0.91245 |

The difference in prediction loss was approximately 0.0432.

This demonstrates that the primer-trained model achieved behavior reasonably close to the reference model on this simple task.

However, GPT 2 did not recover the original dataset, and this experiment does not establish that the total computational cost was reduced. The cost of discovering the primer must also be considered.

## 9. Connection to Cryptanalysis

In classical cryptanalysis, we may observe encrypted information and attempt to infer the unknown key.

In neural network inversion, we observe the trained model weights and attempt to infer information about the training process.

Ordinary neural network training follows:

\[
D_{\text{train}}
\xrightarrow{\text{Training}}
\theta
\]

NIO investigates the inverse direction:

\[
\theta
\xrightarrow{\text{Inverse Optimization}}
D_{\text{synthetic}}
\]

The synthetic dataset may reproduce some learned behavior without being identical to the original training data.

This creates an interesting cybersecurity question:

**How much information about a private training dataset can be extracted from publicly available neural network weights?**

## 10. Student Experiments

**Experiment 1 — Increase Training Steps**

Change:

```python
STEPS = 20
```

to:

```python
STEPS = 40
```

Does NIO still reproduce the reference weights?

**Experiment 2 — Change Primer Size**

Change:

```python
primer = nn.Parameter(torch.randn(6, 3))
```

to:

```python
primer = nn.Parameter(torch.randn(10, 3))
```

Update the primer-window construction to use all available four-character input windows and their next-character targets.

Does a larger primer improve weight matching?

**Experiment 3 — Change the Original Dataset**

Replace:

```python
text = "abc" * 200
```

with:

```python
text = "aabbcc" * 100
```

Can NIO discover a useful primer for a different pattern?

**Experiment 4 — Investigate Model Inversion**

Suppose the original dataset is confidential, but the model weights are publicly available.

Can NIO reveal meaningful information about the structure of the original training data?

How would you distinguish genuine recovery of training information from synthetic data that merely produces similar predictions?

## 11. Research Questions

This small experiment motivates several larger research questions:

- Can NIO discover primers for larger language models?
- Can a synthetic primer reproduce a model's behavior with fewer training steps?
- Can information about private training datasets be recovered from model weights?
- How does primer size affect the quality of weight matching?
- Do different neural network architectures reveal different amounts of training information?
- Can NIO reduce total training cost after accounting for primer optimization?

## 12. Conclusion

This experiment demonstrates how NIO can optimize a small synthetic dataset to reproduce aspects of a trained neural network.

Using a six-position primer, NIO reduced the weight-difference objective substantially and produced a second GPT with prediction performance close to the reference model.

The broader cybersecurity connection is **training-data information leakage through model inversion**.

The experiment does not establish recovery of private training examples, but it provides a simple starting point for investigating what information learned neural network weights may reveal.

The central research question is:

**If we know the architecture and trained weights of a neural network, what can we discover about the data that produced them?**




---





```





import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)

# Training data: predict the next character
text = "abc" * 200
vocab = "abc"
data = torch.tensor([vocab.index(c) for c in text])

# Input sequences of 4 characters
X = torch.stack([data[i:i+4] for i in range(len(data)-4)])
Y = torch.stack([data[i+1:i+5] for i in range(len(data)-4)])

# Tiny GPT-style model
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(3, 8)
        self.attn = nn.MultiheadAttention(8, 1, batch_first=True)
        self.fc = nn.Linear(8, 3)

    def forward(self, x):
        h = self.embed(x)
        mask = torch.triu(
            torch.ones(x.size(1), x.size(1), device=x.device),
            diagonal=1
        ).bool()
        h, _ = self.attn(h, h, h, attn_mask=mask)
        return self.fc(h)

model = TinyGPT()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

losses = []

# Train normally
for step in range(101):
    logits = model(X)
    loss = loss_fn(logits.reshape(-1, 3), Y.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if step % 20 == 0:
        print(f"Step {step}: Loss = {loss.item():.4f}")

plt.plot(losses)
plt.xlabel("Training Step")
plt.ylabel("Cross-Entropy Loss")
plt.title("Tiny GPT Training Without a Primer")
plt.grid(True)
plt.show()






```


attempt 2



```




import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
from torch.func import functional_call

torch.manual_seed(42)

# --------------------------------------------------
# Dataset: Same as Script 1
# --------------------------------------------------

text = "abc" * 200
vocab = "abc"
data = torch.tensor([vocab.index(c) for c in text])

X = torch.stack([data[i:i+4] for i in range(len(data)-4)])
Y = torch.stack([data[i+1:i+5] for i in range(len(data)-4)])


# --------------------------------------------------
# Same TinyGPT architecture
# --------------------------------------------------

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(3, 8)
        self.attn = nn.MultiheadAttention(8, 1, batch_first=True)
        self.fc = nn.Linear(8, 3)

    def forward(self, x):
        if x.ndim == 2:
            h = self.embed(x)
        else:
            h = x @ self.embed.weight

        mask = torch.triu(
            torch.ones(h.size(1), h.size(1), device=h.device),
            diagonal=1
        ).bool()

        h, _ = self.attn(h, h, h, attn_mask=mask)
        return self.fc(h)


# --------------------------------------------------
# GPT 1: Train on original dataset
# --------------------------------------------------

gpt1 = TinyGPT()
initial_weights = copy.deepcopy(gpt1.state_dict())

optimizer = torch.optim.Adam(gpt1.parameters(), lr=0.01)

for step in range(100):
    logits = gpt1(X)
    loss = F.cross_entropy(logits.reshape(-1, 3), Y.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

reference_weights = {
    name: p.detach().clone()
    for name, p in gpt1.named_parameters()
}

print("GPT 1 loss:", loss.item())


# --------------------------------------------------
# GPT 2: NIO primer optimization
# --------------------------------------------------

gpt2 = TinyGPT()
gpt2.load_state_dict(initial_weights)

initial_params = {
    name: p.detach().clone()
    for name, p in gpt2.named_parameters()
}

# Six synthetic characters (not predetermined)
primer = nn.Parameter(torch.randn(6, 3))
nio_optimizer = torch.optim.Adam([primer], lr=0.1)

nio_losses = []


# Train GPT 2 using ONLY the synthetic primer
def train_on_primer(primer_probs):

    px = torch.stack([
        primer_probs[i:i+4] for i in range(2)
    ])

    py = torch.stack([
        primer_probs[i+1:i+5] for i in range(2)
    ])

    # Reset to same initial weights
    params = {
        name: p.clone().requires_grad_(True)
        for name, p in initial_params.items()
    }

    # 20 differentiable training steps
    for step in range(20):

        logits = functional_call(gpt2, params, (px,))

        loss = -(
            py * F.log_softmax(logits, dim=-1)
        ).sum(-1).mean()

        grads = torch.autograd.grad(
            loss,
            tuple(params.values()),
            create_graph=True
        )

        params = {
            name: p - 0.1 * g
            for (name, p), g in zip(params.items(), grads)
        }

    return params


# --------------------------------------------------
# NIO: Optimize ONLY the primer
# --------------------------------------------------

for iteration in range(200):

    probs = F.softmax(primer, dim=-1)

    # Train GPT 2 on primer
    params = train_on_primer(probs)

    # Compare GPT 2 weights to GPT 1
    weight_loss = sum(
        ((params[name] - reference_weights[name]) ** 2).mean()
        for name in params
    )

    nio_optimizer.zero_grad()
    weight_loss.backward()
    nio_optimizer.step()

    nio_losses.append(weight_loss.item())

    if iteration % 20 == 0:
        letters = "".join(
            vocab[i] for i in probs.argmax(dim=1)
        )
        print(
            f"NIO {iteration}: "
            f"Weight Loss = {weight_loss.item():.5f}, "
            f"Primer = {letters}"
        )


# --------------------------------------------------
# Final evaluation
# --------------------------------------------------

final_probs = F.softmax(primer.detach(), dim=-1)

print("\nFinal primer:")
print("".join(
    vocab[i] for i in final_probs.argmax(dim=1)
))

# Retrain GPT 2 from original initialization
# using the final primer ONLY
final_params = train_on_primer(final_probs)

# Evaluate both GPTs on original dataset
# No training on original data for GPT 2
with torch.no_grad():

    logits1 = gpt1(X)
    loss1 = F.cross_entropy(
        logits1.reshape(-1, 3), Y.reshape(-1)
    )

    logits2 = functional_call(gpt2, final_params, (X,))
    loss2 = F.cross_entropy(
        logits2.reshape(-1, 3), Y.reshape(-1)
    )

print("\nFinal comparison:")
print(f"GPT 1 (original data): {loss1.item():.5f}")
print(f"GPT 2 (primer only):   {loss2.item():.5f}")


# --------------------------------------------------
# Plot NIO optimization
# --------------------------------------------------

plt.figure(figsize=(7, 4))
plt.plot(nio_losses)
plt.xlabel("NIO Iteration")
plt.ylabel("Weight Difference")
plt.title("NIO Optimization of GPT Data Primer")
plt.grid(True)
plt.show()


# --------------------------------------------------
# Plot GPT prediction loss comparison
# --------------------------------------------------

plt.figure(figsize=(5, 4))
plt.bar(
    ["GPT 1\nOriginal Data", "GPT 2\nNIO Primer"],
    [loss1.item(), loss2.item()]
)
plt.ylabel("Cross-Entropy Loss")
plt.title("GPT Prediction Loss Comparison")
plt.grid(axis="y")
plt.show()







```


third attempt




```



import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
from torch.func import functional_call

torch.manual_seed(42)

# --------------------------------------------------
# Dataset: Same abc example
# --------------------------------------------------

text = "abc" * 200
vocab = "abc"
data = torch.tensor([vocab.index(c) for c in text])

X = torch.stack([data[i:i+4] for i in range(len(data)-4)])
Y = torch.stack([data[i+1:i+5] for i in range(len(data)-4)])


# --------------------------------------------------
# TinyGPT: Same architecture
# --------------------------------------------------

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(3, 8)
        self.attn = nn.MultiheadAttention(8, 1, batch_first=True)
        self.fc = nn.Linear(8, 3)

    def forward(self, x):
        if x.ndim == 2:
            h = self.embed(x)
        else:
            h = x @ self.embed.weight

        mask = torch.triu(
            torch.ones(h.size(1), h.size(1), device=h.device),
            diagonal=1
        ).bool()

        h, _ = self.attn(h, h, h, attn_mask=mask)
        return self.fc(h)


# --------------------------------------------------
# Common initialization and training settings
# --------------------------------------------------

gpt1 = TinyGPT()
initial_weights = copy.deepcopy(gpt1.state_dict())

STEPS = 20
LR = 0.1


# --------------------------------------------------
# GPT 1: Train on original dataset using SGD
# --------------------------------------------------

optimizer = torch.optim.SGD(gpt1.parameters(), lr=LR)

for step in range(STEPS):

    logits = gpt1(X)
    loss = F.cross_entropy(
        logits.reshape(-1, 3), Y.reshape(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

reference_weights = {
    name: p.detach().clone()
    for name, p in gpt1.named_parameters()
}

print("GPT 1 loss:", loss.item())


# --------------------------------------------------
# GPT 2: Same initial weights
# --------------------------------------------------

gpt2 = TinyGPT()
gpt2.load_state_dict(initial_weights)

initial_params = {
    name: p.detach().clone()
    for name, p in gpt2.named_parameters()
}


# --------------------------------------------------
# NIO Primer: Six synthetic characters
# --------------------------------------------------

primer = nn.Parameter(torch.randn(6, 3))
nio_optimizer = torch.optim.Adam([primer], lr=0.1)

nio_losses = []


# --------------------------------------------------
# Train GPT 2 using ONLY the primer
# --------------------------------------------------

def train_on_primer(primer_probs):

    px = torch.stack([
        primer_probs[i:i+4] for i in range(2)
    ])

    py = torch.stack([
        primer_probs[i+1:i+5] for i in range(2)
    ])

    # Reset GPT 2 to original weights
    params = {
        name: p.clone().requires_grad_(True)
        for name, p in initial_params.items()
    }

    # Same 20 SGD steps as GPT 1
    for step in range(STEPS):

        logits = functional_call(gpt2, params, (px,))

        loss = -(
            py * F.log_softmax(logits, dim=-1)
        ).sum(-1).mean()

        grads = torch.autograd.grad(
            loss,
            tuple(params.values()),
            create_graph=True
        )

        params = {
            name: p - LR * g
            for (name, p), g in zip(params.items(), grads)
        }

    return params


# --------------------------------------------------
# NIO: Optimize ONLY the primer
# --------------------------------------------------

for iteration in range(200):

    probs = F.softmax(primer, dim=-1)

    params = train_on_primer(probs)

    weight_loss = sum(
        ((params[name] - reference_weights[name]) ** 2).mean()
        for name in params
    )

    nio_optimizer.zero_grad()
    weight_loss.backward()
    nio_optimizer.step()

    nio_losses.append(weight_loss.item())

    if iteration % 20 == 0:
        letters = "".join(
            vocab[i] for i in probs.argmax(dim=1)
        )

        print(
            f"NIO {iteration}: "
            f"Weight Loss = {weight_loss.item():.5f}, "
            f"Primer = {letters}"
        )


# --------------------------------------------------
# Final evaluation
# --------------------------------------------------

final_probs = F.softmax(primer.detach(), dim=-1)

print("\nFinal primer:")
print("".join(
    vocab[i] for i in final_probs.argmax(dim=1)
))

# Retrain GPT 2 using final primer only
final_params = train_on_primer(final_probs)

# Evaluate both models on original dataset
with torch.no_grad():

    logits1 = gpt1(X)
    loss1 = F.cross_entropy(
        logits1.reshape(-1, 3), Y.reshape(-1)
    )

    logits2 = functional_call(gpt2, final_params, (X,))
    loss2 = F.cross_entropy(
        logits2.reshape(-1, 3), Y.reshape(-1)
    )

print("\nFinal comparison:")
print(f"GPT 1 (original data): {loss1.item():.5f}")
print(f"GPT 2 (primer only):   {loss2.item():.5f}")


# --------------------------------------------------
# Plot 1: NIO optimization
# --------------------------------------------------

plt.figure(figsize=(7, 4))
plt.plot(nio_losses)
plt.xlabel("NIO Iteration")
plt.ylabel("Weight Difference")
plt.title("NIO Optimization of GPT Data Primer")
plt.grid(True)
plt.show()


# --------------------------------------------------
# Plot 2: GPT prediction loss comparison
# --------------------------------------------------

plt.figure(figsize=(5, 4))
plt.bar(
    ["GPT 1\nOriginal Data", "GPT 2\nNIO Primer"],
    [loss1.item(), loss2.item()]
)
plt.ylabel("Cross-Entropy Loss")
plt.title("GPT Prediction Loss Comparison")
plt.grid(axis="y")
plt.show()






```













