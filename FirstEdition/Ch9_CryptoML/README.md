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
