## NIO Dataset Primer







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













