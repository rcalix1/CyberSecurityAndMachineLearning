## Model Extraction

* or model stealing
* Stealing Machine Learning Models via Prediction APIs
* https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/tramer

---


# Model Extraction or Model Stealing

Model stealing refers to the process of extracting information from a trained model and using that information to train another model.

This has attracted attention because of reports and allegations that AI labs have queried powerful language models to obtain outputs, including reasoning traces, that could be used to train competing models.

The important idea is that **the attacker does not need access to the original model's weights or training data**.

Instead, the attacker sends inputs to the original model and collects its outputs.

## 1. The Original Model

Assume we have an original trained model:

$$
y=f_{\theta}(x)
$$

where:

- $x$ represents the input.
- $y$ represents the output.
- $f$ represents the original model.
- $\theta$ represents the trained weights of the original model.

The attacker does not have access to the weights $\theta$.

However, the attacker can query the model by providing inputs and observing the outputs.

$$
\hat{y}_i=f_{\theta}(x_i)
$$

For example, an attacker might submit 500 different inputs and record the model's predictions.

## 2. Construct a New Dataset

Given the inputs and the responses from the original model, the attacker can construct a new dataset:

$$
D_{\text{steal}}=\{(x_i,\hat{y}_i)\}_{i=1}^{N}
$$

Here:

- $D_{\text{steal}}$ is the new dataset constructed from queries.
- $x_i$ is an input submitted to the original model.
- $\hat{y}_i$ is the original model's response.
- $N$ is the total number of queries.

Notice that the attacker is not stealing the original training dataset directly.

Instead, the attacker is creating a new dataset containing the original model's input-output behavior.

## 3. Train the Stolen Model

The attacker now trains another model:

$$
g_{\phi}(x)
$$

where $\phi$ represents the weights of the new model.

The goal is to make the new model behave similarly to the original model:

$$
g_{\phi}(x_i)\approx f_{\theta}(x_i)
$$

The attacker may therefore obtain a model that approximates the original model's behavior at a fraction of its original training cost.

Importantly, the attacker does not necessarily recover the original weights. The goal is to reproduce the model's predictions.

## 4. Python Example: Model Extraction

The following code demonstrates the complete process.

We first train an original model, `model_A`.

Next, we simulate an attacker querying that model to obtain predictions.

Finally, we train `model_B` using those predictions and compare the two models.

```python
import torch
import torch.nn as nn

torch.manual_seed(1)

# --------------------------------------------------
# STEP 1: Train the original model (Model A)
# --------------------------------------------------

X = torch.randn(200, 2)

y = (X[:, 0] + X[:, 1] > 0).long()

model_A = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 2)
)

optimizer = torch.optim.Adam(model_A.parameters(), lr=0.05)
loss_fn = nn.CrossEntropyLoss()

for i in range(200):

    output = model_A(X)
    loss = loss_fn(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# --------------------------------------------------
# STEP 2: Attack begins here
# --------------------------------------------------

# Attacker generates new inputs.
X_attack = torch.randn(500, 2)

# Attacker queries Model A to obtain its predictions.
# The attacker does not access Model A's weights.

with torch.no_grad():
    stolen_labels = model_A(X_attack).argmax(dim=1)


# --------------------------------------------------
# STEP 3: Train Model B using the stolen predictions
# --------------------------------------------------

model_B = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 2)
)

optimizer = torch.optim.Adam(model_B.parameters(), lr=0.05)

for i in range(200):

    output = model_B(X_attack)
    loss = loss_fn(output, stolen_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# --------------------------------------------------
# STEP 4: Compare Model A and Model B
# --------------------------------------------------

X_test = torch.randn(1000, 2)

with torch.no_grad():

    A = model_A(X_test).argmax(dim=1)
    B = model_B(X_test).argmax(dim=1)

agreement = (A == B).float().mean()

print("Model A / Model B agreement:", agreement.item())
```

### Example Output

```text
Model A / Model B agreement: 0.9990000128746033
```

The example output indicates approximately 99.9% agreement between the two models.

This means that Model B produces the same classifications as Model A for approximately 999 out of 1,000 test inputs.

The exact agreement may vary with training conditions.

**Important:** Agreement measures how closely Model B imitates Model A. It does not necessarily mean either model achieves 99.9% accuracy against the true labels.

## 5. What Did the Attacker Actually Steal?

Notice that the attacker never copied the original model's weights.

Instead, the attacker collected predictions:

```python
stolen_labels = model_A(X_attack).argmax(dim=1)
```

These predictions became the training labels for Model B:

```python
loss = loss_fn(output, stolen_labels)
```

This is the central idea behind model extraction.

The attacker learns to reproduce the behavior of the original model by observing its responses.

## 6. Connection to Federated Learning and Privacy

Federated learning and model extraction illustrate two different aspects of machine learning privacy and security.

**Federated Learning:**

- Training data remains with individual clients.
- Clients share model updates rather than their raw datasets.
- Additional mechanisms may be required to protect sensitive information.

**Model Extraction:**

- An attacker may only have access to a model's inputs and outputs.
- The attacker collects predictions from the original model.
- Those predictions are used to train a substitute model.
- The substitute model may reproduce the original model's behavior without recovering its weights.

### Main Takeaway

**Keeping training data private does not automatically prevent someone from learning information about a trained model.**

Model extraction demonstrates that a model's predictions can themselves be valuable information.

This is an important consideration when deploying machine learning models through public APIs or other query interfaces.











---
