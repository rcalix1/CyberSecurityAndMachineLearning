## LLM and Cyber Security

* Adversarial Text Embeddings

---


## Adversarial Machine Learning: Attacking Word Embeddings

### 1. Basic Idea

A language model converts input tokens into numerical vectors called **embeddings**. These vectors are processed by a neural network to make predictions.

An adversarial attack introduces small perturbations into these embeddings to change the model's predictions.

For example, our sentiment classifier initially predicts:

**Input:** "I really like the movie."

**Original prediction:** Positive (class 1)

**After perturbation:** Negative (class 0)

The input sentence remains unchanged. Only the numerical embeddings are modified.

### 2. Mathematical Representation

An embedding layer is a matrix:

$$
E \in \mathbb{R}^{V \times d}
$$

where:

- $V$ = vocabulary size (30,522 tokens)
- $d$ = embedding dimension (768)
- $E$ = embedding matrix

Each token $i$ is represented by a vector:

$$
\mathbf{e}_i = E[i]
$$

For a sentence containing $n$ tokens, the model retrieves:

$$
X = [\mathbf{e}_1,\mathbf{e}_2,\ldots,\mathbf{e}_n]
$$

In our example, the tokenized sentence contains 8 tokens:

$$
X \in \mathbb{R}^{8 \times 768}
$$

The model predicts the sentiment using:

$$
\hat{y} = \arg\max_k f_\theta(X)_k
$$

where $f_\theta$ is the neural network and $k$ represents the output class.

### 3. Adversarial Perturbation

We introduce random Gaussian noise into the embedding matrix:

$$
E_{\text{adv}} = E + \Delta
$$

where:

$$
\Delta = \epsilon Z
$$

and:

$$
Z_{ij} \sim \mathcal{N}(0,1)
$$

Here:

- $\Delta$ = adversarial perturbation matrix
- $\epsilon$ = perturbation strength
- $Z$ = random matrix sampled from a standard normal distribution

In Python:

```python
epsilon = 0.1

perturbation = epsilon * torch.randn_like(embeddings_tr)

adversarial_embeddings = embeddings_tr + perturbation

embedding_layer.weight.data = adversarial_embeddings
```

When $\epsilon=0$, the embeddings remain unchanged.

As $\epsilon$ increases, the embedding vectors move away from their original positions.

**Important:** This experiment uses random perturbations rather than an optimized adversarial attack. It demonstrates the sensitivity of a neural network to changes in its embeddings.

### 4. Effect on Classification

Before perturbation, the classifier produces:

$$
f_\theta(X) = [-4.1794,\;4.4663]
$$

Therefore:

$$
\hat{y} = \arg\max([-4.1794,4.4663]) = 1
$$

**Prediction: Positive**

After perturbing the embedding matrix:

$$
f_{\theta,\text{adv}}(X) = [2.7225,\;-2.2167]
$$

Therefore:

$$
\hat{y}_{\text{adv}} = \arg\max([2.7225,-2.2167]) = 0
$$

**Prediction: Negative**

The original sentence has not changed, yet the classifier produces the opposite result.

### 5. From Random Noise to Adversarial Optimization

A more sophisticated attack deliberately searches for a perturbation that maximizes the classification error:

$$
\Delta^* = \arg\max_{\|\Delta\|\leq\epsilon}
\mathcal{L}(f_\theta(X+\Delta),y)
$$

where:

- $\mathcal{L}$ = classification loss function
- $y$ = correct class
- $\epsilon$ = maximum allowed perturbation magnitude

One popular method is the **Fast Gradient Sign Method (FGSM)**:

$$
\Delta = \epsilon \operatorname{sign}
\left(\nabla_X \mathcal{L}(f_\theta(X),y)\right)
$$

FGSM uses gradients to identify directions that increase the model's classification loss.

Unlike random noise, FGSM deliberately pushes the input toward an incorrect prediction.

### 6. Key Observation

This experiment illustrates three important concepts:

1. **Embedding sensitivity:** Small numerical changes can influence neural-network predictions.
2. **Prediction instability:** A classifier may produce different results even when the original text remains unchanged.
3. **Adversarial optimization:** Gradients can be used to construct perturbations that deliberately cause classification errors.

**Note:** The provided experiment modifies the entire embedding matrix, including embeddings for tokens not present in the sentence. It is therefore a model-parameter perturbation experiment rather than a conventional input-only adversarial attack. Also, $\epsilon=0.1$ controls the noise standard deviation, not a strict maximum perturbation size.

### Reference

Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). *Explaining and Harnessing Adversarial Examples.* International Conference on Learning Representations (ICLR).

https://arxiv.org/abs/1412.6572








---
