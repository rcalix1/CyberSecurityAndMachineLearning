
# Machine Learning Privacy and Security — Part 2

## Four Simple Experiments (80-Minute Lecture)

This lecture continues our discussion of machine learning privacy and security using four simple experiments.

We will investigate membership inference, model inversion, model stealing, and data drift.

Each experiment uses Python and PyTorch and requires no advanced mathematics.

## Lecture Schedule

| Topic | Time | Main Question |
|---|---|---|
| 1. Membership Inference | 20 min | Can we determine whether a sample was used to train a model? |
| 2. Model Inversion | 20 min | Can we reconstruct an input by optimizing it against a frozen model? |
| 3. Model Stealing | 20 min | How many queries are needed to copy a model? |
| 4. Data Drift | 20 min | How does changing the data distribution affect accuracy? |
| **Total** | **80 min** | |

---

## 1. Membership Inference (20 Minutes)

Membership inference is an attack that attempts to determine whether a particular sample was used to train a machine learning model.

The basic idea is that a model may be more confident when predicting samples it has already seen during training.

Suppose a model returns the following probabilities:

- Training sample: 99% confidence.
- Unknown sample: 60% confidence.

An attacker might guess that the first sample was part of the training dataset.

However, confidence alone does not prove membership.

### Python Example

We will train a simple neural network and compare its confidence on training data and new data.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# Generate training data with random labels.
# Random labels encourage memorization.

X_train = torch.randn(40, 2)
y_train = torch.randint(0, 2, (40,))

# Create a neural network.
model = nn.Sequential(
    nn.Linear(2, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# Train the model.
for epoch in range(500):

    output = model(X_train)
    loss = loss_fn(output, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Generate new samples.
X_new = torch.randn(40, 2)

# Compare confidence.
with torch.no_grad():

    train_probs = F.softmax(model(X_train), dim=1)
    new_probs = F.softmax(model(X_new), dim=1)

    train_confidence = train_probs.max(dim=1).values.mean()
    new_confidence = new_probs.max(dim=1).values.mean()

print("Training confidence:", train_confidence.item())
print("New data confidence:", new_confidence.item())
```

### Understanding the Experiment

The model has only 40 training samples, but it has enough parameters to memorize them.

We compare the average maximum softmax probability:

$$
C(x)=\max_k P(y=k\mid x)
$$

where $C(x)$ is the model's confidence in its most likely class.

If training confidence is significantly higher than confidence on new samples, this may indicate that the model is memorizing its training data.

This is a simple confidence-based membership inference demonstration, not a reliable membership classifier by itself.

### Student Experiment

Change the number of training samples from 40 to 200, then 1,000.

Does the difference between training confidence and new-data confidence become smaller?

---

## 2. Model Inversion (20 Minutes)

Model inversion attempts to reconstruct information about a model's inputs by using the trained model.

Instead of training the model's weights, we freeze them and optimize the input.

This is an example of an inverse problem.

### Mathematical Idea

Normally, a neural network calculates:

$$
y=f_\theta(x)
$$

where $x$ is the input and $\theta$ represents the model's trained parameters.

In model inversion, we know a desired output $y^*$ and want to find an input $x$ that produces it.



$$
x^* = \arg\min_{x} L(f_{\theta}(x), y^*)
$$

The model weights remain fixed.


The model weights remain fixed.

Only the input changes.

### Python Example

```python
import torch
import torch.nn as nn

torch.manual_seed(1)

# Create a simple target function.
X = torch.randn(200, 2)

y = (X[:, 0] + X[:, 1]).unsqueeze(1)

# Train a neural network.
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(500):

    prediction = model(X)
    loss = loss_fn(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ---------------------------------------
# MODEL INVERSION BEGINS
# ---------------------------------------

# Freeze the model.
for param in model.parameters():
    param.requires_grad = False

# Desired output.
target = torch.tensor([[5.0]])

# Make the INPUT trainable.
x_inverse = torch.randn(1, 2, requires_grad=True)

optimizer = torch.optim.Adam([x_inverse], lr=0.05)

for step in range(300):

    prediction = model(x_inverse)

    loss = loss_fn(prediction, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Recovered input:", x_inverse.detach())
print("Model output:", model(x_inverse).detach())
```

### Understanding the Experiment

Notice that the neural network weights are frozen:

```python
for param in model.parameters():
    param.requires_grad = False
```

Instead, the input is trainable:

```python
x_inverse = torch.randn(1, 2, requires_grad=True)
```

The optimizer updates the input:

```python
optimizer = torch.optim.Adam([x_inverse], lr=0.05)
```

The goal is to find an input that makes the model produce an output close to 5.

Since the original relationship is approximately:

$$
y=x_1+x_2
$$

there are many possible solutions:

$$
2+3=5
$$

$$
1+4=5
$$

$$
-5+10=5
$$

**Important:** Model inversion does not necessarily recover the exact original input. It may recover one of many inputs producing the desired output.

### Student Experiment

Change the desired output from 5 to 10, then -3.

Can the optimization find inputs producing these outputs?

---

## 3. Model Stealing: How Many Queries Are Needed? (20 Minutes)

Previously, we demonstrated how an attacker can train a new model using predictions from another model.

Now we investigate a different question:

**How many queries are needed to reproduce the original model's behavior?**

### Mathematical Idea

The original model is:

$$
y=f_\theta(x)
$$

The attacker queries the original model and collects predictions:

$$
D_{\text{steal}}=\{(x_i,f_\theta(x_i))\}_{i=1}^{N}
$$

A new model is trained to imitate the original model:

$$
g_\phi(x)\approx f_\theta(x)
$$

We will vary $N$, the number of queries.

### Python Example

```python
import torch
import torch.nn as nn

torch.manual_seed(1)

# ---------------------------------------
# TRAIN ORIGINAL MODEL A
# ---------------------------------------

X = torch.randn(200, 2)
y = (X[:, 0] + X[:, 1] > 0).long()

model_A = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 2)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_A.parameters(), lr=0.05)

for epoch in range(200):

    output = model_A(X)
    loss = loss_fn(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Fixed test set for comparison.
X_test = torch.randn(1000, 2)

with torch.no_grad():
    A = model_A(X_test).argmax(dim=1)

# ---------------------------------------
# MODEL STEALING EXPERIMENT
# ---------------------------------------

query_counts = [10, 20, 50, 100, 500, 1000]

for n_queries in query_counts:

    # Attacker generates queries.
    X_attack = torch.randn(n_queries, 2)

    # Obtain predictions from Model A.
    with torch.no_grad():
        stolen_labels = model_A(X_attack).argmax(dim=1)

    # Create Model B.
    model_B = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    )

    optimizer = torch.optim.Adam(
        model_B.parameters(),
        lr=0.05
    )

    # Train Model B.
    for epoch in range(200):

        output = model_B(X_attack)
        loss = loss_fn(output, stolen_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Compare predictions.
    with torch.no_grad():

        B = model_B(X_test).argmax(dim=1)

        agreement = (A == B).float().mean().item()

    print(
        f"Queries: {n_queries:4d} | "
        f"Agreement: {agreement:.2%}"
    )
```

### Understanding the Experiment

The attacker never accesses Model A's weights.

Instead, the attacker collects its predictions and uses them to train Model B.

We calculate agreement:

$$
\text{Agreement}=
\frac{\text{Number of matching predictions}}
{\text{Total test predictions}}
$$

For example, if both models agree on 950 out of 1,000 samples:

$$
\text{Agreement}=\frac{950}{1000}=95\%
$$

Agreement is not the same as classification accuracy.

Two models can agree with each other while both make incorrect predictions.

### Student Experiment

Observe agreement for different query counts.

Does increasing the number of queries always improve agreement?

Why might a simple model be easier to imitate than a more complex model?

---

## 4. Data Drift: Does Drift Reduce Accuracy? (20 Minutes)

Machine learning models are trained using data from a particular distribution.

Over time, the data distribution may change.

This is called data drift.

We will examine how changes in the input distribution affect a trained model.

### Mathematical Idea

Suppose training data follows distribution $P$ and new data follows distribution $Q$.

KL divergence measures the difference between the two distributions:

$$
D_{KL}(P\|Q)=
\sum_x P(x)\ln\left(\frac{P(x)}{Q(x)}\right)
$$

For this experiment, we use a simple Gaussian distribution.

The KL divergence between two one-dimensional Gaussian distributions with the same variance is:

$$
D_{KL}(P\|Q)=
\frac{(\mu_P-\mu_Q)^2}{2\sigma^2}
$$

where:

- $\mu_P$ is the original mean.
- $\mu_Q$ is the new mean.
- $\sigma$ is the shared standard deviation.

### Python Example

```python
import torch
import torch.nn as nn

torch.manual_seed(1)

# ---------------------------------------
# TRAIN MODEL ON ORIGINAL DATA
# ---------------------------------------

X_train = torch.randn(1000, 2)

y_train = (X_train[:, 0] + X_train[:, 1] > 0).long()

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 2)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(200):

    output = model(X_train)
    loss = loss_fn(output, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ---------------------------------------
# INTRODUCE DATA DRIFT
# ---------------------------------------

shifts = [0, 0.5, 1, 2, 3, 5]

# Use the same underlying test samples
# to isolate the effect of shifting the data.
X_base = torch.randn(1000, 2)

for shift in shifts:

    # Shift both features.
    X_test = X_base + shift

    # Calculate true labels using the
    # original classification rule.
    y_test = (
        X_test[:, 0] + X_test[:, 1] > 0
    ).long()

    with torch.no_grad():

        preds = model(X_test).argmax(dim=1)

        accuracy = (
            (preds == y_test).float().mean().item()
        )

    # KL divergence for two independent
    # Gaussian features, variance = 1.
    kl = shift ** 2

    print(
        f"Shift: {shift:3.1f} | "
        f"KL: {kl:6.2f} | "
        f"Accuracy: {accuracy:.2%}"
    )
```

### Understanding the Experiment

The original training data is generated using:

```python
X_train = torch.randn(1000, 2)
```

This produces normally distributed data with mean zero and standard deviation one.

We then introduce drift:

```python
X_test = X_base + shift
```

For example:

- `shift = 0`: Original distribution.
- `shift = 1`: Distribution shifted by 1.
- `shift = 3`: Distribution shifted by 3.
- `shift = 5`: Distribution shifted by 5.

We calculate both KL divergence and classification accuracy.

The classification rule remains unchanged:

$$
y=
\begin{cases}
1 & x_1+x_2>0\\
0 & x_1+x_2\leq 0
\end{cases}
$$

### Important Observation

Increasing data drift does not necessarily reduce accuracy.

A model may continue making correct predictions even when the input distribution changes significantly.

In this example, larger positive shifts may actually make many samples easier to classify because they move farther from the decision boundary.

This demonstrates that **data drift and model performance degradation are related but different concepts**.

### Student Experiment

Try negative shifts:

```python
shifts = [-5, -3, -1, 0, 1, 3, 5]
```

Does the model perform equally well for positive and negative shifts?

Can you find a shift that produces high KL divergence without substantially reducing accuracy?

---

## Final Discussion

These four experiments illustrate important machine learning security concepts.

| Topic | What We Learned |
|---|---|
| Membership Inference | Model confidence may reveal information about training membership. |
| Model Inversion | Inputs can be optimized against a frozen neural network. |
| Model Stealing | A substitute model can learn to imitate another model through queries. |
| Data Drift | Changes in input distributions do not always imply lower accuracy. |

### Key Takeaway

Protecting machine learning systems requires more than simply keeping the original training data private.

Models can reveal information through their predictions, can be imitated by other models, and may behave differently when deployed on changing data distributions.

These experiments provide simple ways to explore those issues using PyTorch.
