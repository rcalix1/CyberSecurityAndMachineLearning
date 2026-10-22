## ML and Crypto

* link

## A neural net

```python


# !pip install torch tenseal

import torch
import tenseal as ts

# 1. Simple artificial data: y = x1 + x2
X = torch.tensor([
    [1., 2.],
    [2., 3.],
    [3., 4.],
    [4., 5.]
])

Y = torch.tensor([
    [3.],
    [5.],
    [7.],
    [9.]
])

# 2. Neural network: 2 inputs -> 1 output
model = torch.nn.Linear(2, 1)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(2000):
    optimizer.zero_grad()
    loss = ((model(X) - Y)**2).mean()
    loss.backward()
    optimizer.step()

print("Weights:", model.weight.data)
print("Bias:", model.bias.data)

# 3. Create encryption context
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)

context.global_scale = 2**40
context.generate_galois_keys()

# 4. Encrypt new input [5, 6]
x = [5., 6.]
enc_x = ts.ckks_vector(context, x)

# 5. Neural network inference on encrypted data
W = model.weight.detach().numpy().T.tolist()
b = model.bias.item()

enc_y = enc_x.matmul(W) + b

# 6. Decrypt prediction
print("Encrypted NN prediction:", enc_y.decrypt())
print("Expected answer:", 11)



```


---




## Simplified CKKS Encryption: Numerical Example

**The key idea:** Encryption is designed so that the secret-key terms cancel during decryption. This also allows computations on encrypted values.

### 1. Generate Keys

Choose:

- Secret key: $s=3$
- Random number: $a=7$
- Small error: $e=1$
- Scale: $\Delta=1000$

Compute the public key component:

$$
b=-as+e=-7(3)+1=-20
$$

Public key: $(a,b)=(7,-20)$.

### 2. Encrypt the Number 5

Choose a random value $u=2$.

The simplified encryption equations are:

$$
c_0=bu+m\Delta,\qquad c_1=au
$$

Substitute:

$$
c_0=-20(2)+5(1000)=4960
$$

$$
c_1=7(2)=14
$$

Encrypted message:

$$
\boxed{\operatorname{Enc}(5)=(4960,14)}
$$

### 3. Decrypt: Why Does It Work?

Decryption uses the secret key $s=3$:

$$
m'=\frac{c_0+c_1s}{\Delta}
$$

Substitute:

$$
m'=\frac{4960+14(3)}{1000}
=\boxed{5.002}
$$

**Why does this recover the original number?**

Substitute the encryption equations into decryption:

$$
c_0+c_1s=(-as+e)u+m\Delta+aus
$$

The secret-key terms cancel:

$$
\underbrace{-asu+aus}_{0}+eu+m\Delta
$$

Therefore:

$$
\boxed{m'=m+\frac{eu}{\Delta}}
$$

The original number is recovered with a small approximation error.

### 4. Why Can We Add Encrypted Numbers?

Encrypt 6 using $u=3$:

$$
c_0=-20(3)+6(1000)=5940
$$

$$
c_1=7(3)=21
$$

Therefore:

$$
\operatorname{Enc}(6)=(5940,21)
$$

Add the encrypted pairs without decrypting:

$$
(4960,14)+(5940,21)=(10900,35)
$$

Decrypt only the final result:

$$
m'=\frac{10900+35(3)}{1000}
$$

$$
\boxed{m'=11.005\approx 5+6}
$$

**Why does this work?**

Because the secret-key terms still cancel when ciphertexts are added, leaving the sum of the original messages and their small errors.

### 5. Connection to Neural Networks

A neural network computes:

$$
y=w_1x_1+w_2x_2+b
$$

CKKS allows these additions and multiplications to be performed on encrypted inputs, without first decrypting them.

**Note:** This is an educational simplification using ordinary integers. Real CKKS uses polynomial rings, modular arithmetic, additional randomness, and more sophisticated operations for multiplication.












