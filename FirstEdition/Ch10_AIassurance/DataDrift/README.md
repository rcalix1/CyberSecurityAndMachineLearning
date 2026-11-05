## Data Drift 


# Data or Model Drift

Data or model drift relates to the problem of how data can change over time.

Initially, we may train a machine learning model using data that follows a particular probability distribution, which we will call $P$.

Over time, the data distribution may change. The new distribution can be represented as $Q$.

For example:

- **Original distribution ($P$):** Data used to train the model.
- **New distribution ($Q$):** Data collected after the model has been deployed.

If the distributions become sufficiently different, the model may no longer perform as well as it did during training.

This is known as **data drift**.

Model drift is a related concept involving changes in model behavior or performance. Data drift can contribute to model performance degradation, but the two are not necessarily the same.

## 1. Measuring Drift with KL Divergence

A well-known metric used to measure differences between probability distributions is the **Kullback-Leibler (KL) divergence**, from Information Theory.

KL divergence measures how different one probability distribution is from another.

The formula is:

$$
D_{KL}(P\|Q)=\sum_x P(x)\ln\left(\frac{P(x)}{Q(x)}\right)
$$

where:

- $P(x)$ is the probability of an event under distribution $P$.
- $Q(x)$ is the probability of the same event under distribution $Q$.
- $\ln$ is the natural logarithm.
- $D_{KL}$ is the KL divergence between the distributions.

### Interpretation

- $D_{KL}=0$: The distributions are identical.
- Small KL divergence: The distributions are relatively similar.
- Large KL divergence: The distributions are more different.

KL divergence is not symmetric:

$$
D_{KL}(P\|Q)\neq D_{KL}(Q\|P)
$$

This means the order of the distributions matters.

For datasets containing many features, one simple approach is to calculate KL divergence separately for each feature, after estimating its probability distribution.

## 2. Calculating KL Divergence in PyTorch

PyTorch provides a convenient function for calculating KL divergence:

```python
nn.KLDivLoss()
```

The function expects:

- `input`: Log probabilities.
- `target`: Probabilities.

By default, the function computes the divergence from the target distribution to the distribution represented by `input`.

We will generate three probability distributions and compare them.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

kl_loss = nn.KLDivLoss(reduction="batchmean")

# --------------------------------------------------
# STEP 1: Generate the original distribution
# --------------------------------------------------

input = F.log_softmax(
    torch.randn(9, 8, requires_grad=True),
    dim=1
)

print(input.shape)

# torch.Size([9, 8])


# --------------------------------------------------
# STEP 2: Generate the first target distribution
# --------------------------------------------------

target = F.softmax(
    torch.rand(9, 8),
    dim=1
)

print(target.shape)

# torch.Size([9, 8])


# --------------------------------------------------
# STEP 3: Calculate KL divergence
# --------------------------------------------------

output = kl_loss(input, target)

print(output)

# Example:
# tensor(0.4355, grad_fn=<DivBackward0>)


# --------------------------------------------------
# STEP 4: Generate a second target distribution
# --------------------------------------------------

tensor2 = torch.rand(9, 8) * 3 + 2

target2 = F.softmax(tensor2, dim=1)

print(target2.shape)

# torch.Size([9, 8])


# --------------------------------------------------
# STEP 5: Calculate KL divergence again
# --------------------------------------------------

output = kl_loss(input, target2)

print(output)

# Example:
# tensor(0.7253, grad_fn=<DivBackward0>)
```

## 3. Understanding the Results

In this example, we compare the original input distribution with two target distributions.

| Comparison | Example KL Divergence |
|---|---:|
| `input` vs. `target` | 0.4355 |
| `input` vs. `target2` | 0.7253 |

The second comparison produces a larger KL divergence.

$$
0.7253 > 0.4355
$$

This means that, for this particular execution, `target2` is more different from the input probability distribution than `target`.

**Important:** These distributions are randomly generated, so the results will change each time the code runs. A larger divergence for `target2` is not guaranteed.

Also, adding 2 to every element of `tensor2` does not affect its softmax probabilities. The multiplication by 3 is what changes their relative concentration.

## 4. Why Is This Useful for Machine Learning?

Imagine training a neural network using network traffic collected from a company's network.

During training, the network traffic follows a particular distribution.

Several months later, the company changes its network infrastructure, introduces new applications, or experiences new types of traffic.

The distribution of incoming network traffic may now be different.

We can compare the original training distribution with the new data distribution:

$$
D_{KL}(P_{\text{training}}\|P_{\text{new}})
$$

If the divergence increases, it may indicate that the data has changed significantly.

This could be a reason to investigate the model's performance and potentially retrain it using more recent data.

However, a high KL divergence does not automatically mean that the model's accuracy has decreased. Model performance must be evaluated separately.

## 5. Main Takeaway

**Machine learning models are trained using data from particular probability distributions, but real-world data can change over time.**

KL divergence provides a mathematical way to measure differences between probability distributions.

By monitoring these differences, we can identify potential data drift and determine when further model evaluation or retraining may be necessary.

