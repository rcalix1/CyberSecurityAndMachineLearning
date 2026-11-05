## Federated Learning

* Communication-Efficient Learning of Deep Networks from Decentralized Data
* https://proceedings.mlr.press/v54/mcmahan17a.html

# Federated Learning and Privacy

In this section, we will cover how to maintain privacy using **Federated Learning**.

In federated learning, we have a **global model** and several **local models**. Each local model is trained separately using its own data. The parameters of these local models are then averaged to update the global model.

An important advantage is that the original training data stays with each client rather than being collected in one central location.

We will also demonstrate adding random noise to the averaged model parameters as an additional privacy mechanism.

**Note:** Adding noise can help obscure information, but this simple example does not provide a formal differential privacy guarantee.

## 1. Define the Neural Network

We begin by importing the libraries and defining a very simple classification neural network.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()

        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x):

        return self.net(x)
```

The neural network consists of a single linear layer. It receives input features and produces classification scores.

## 2. Generate Random Client Data

For this example, we will generate random data for each client.

Each client has its own dataset containing the familiar `X` (inputs) and `y` (labels).

```python
def generate_client_data(n_clients, samples_per_client, input_dim):
    clients_data = []

    for _ in range(n_clients):

        data = np.random.randn(samples_per_client, input_dim)

        labels = (data.sum(axis=1) > 0).astype(int)

        clients_data.append(
            (
                torch.tensor(data, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.long)
            )
        )

    return clients_data
```

The classification rule is simple:

- If the sum of the input features is greater than zero, the label is `1`.
- Otherwise, the label is `0`.

Each client receives its own randomly generated training data.

## 3. Train the Local Models

Next, we define our familiar neural network training function.

```python
def train_local_model(model, xs, ys, epochs, lr):

    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for i in range(epochs):

        optimizer.zero_grad()

        outputs = model(xs)

        loss = criterion(outputs, ys)

        loss.backward()

        optimizer.step()

    return model.state_dict()
```

Each client trains its own copy of the neural network using its local data.

The function returns the trained model parameters, rather than the original training data.

## 4. Average Local Models and Add Noise

This is where the main federated learning operation happens.

After the local models have been trained, we combine their parameters by calculating their mean.

### Step 1: Local Model Parameters

Suppose we have $K$ clients.

The parameters of their local models can be represented as:

$$
\theta_1,\theta_2,\ldots,\theta_K
$$

Here:

- $K$ is the number of clients.
- $\theta_k$ represents the parameters of client $k$.

### Step 2: Average the Parameters

The local models are combined by taking the mean of their parameters:

$$
\bar{\theta}=\frac{1}{K}\sum_{k=1}^{K}\theta_k
$$

For example, suppose three clients have a particular weight equal to:

$$
\theta_1=2,\quad \theta_2=4,\quad \theta_3=6
$$

The average is:

$$
\bar{\theta}=\frac{2+4+6}{3}=4
$$

The global model receives this averaged weight.

In practice, this calculation is performed for every model parameter.

### Step 3: Add Gaussian Noise

To help obscure information about individual client updates, we add random noise.

The noise follows a normal distribution:

$$
\epsilon\sim\mathcal{N}(0,\sigma^2)
$$

Here:

- $\epsilon$ represents random noise.
- $\sigma$ controls the amount of noise added.
- Larger $\sigma$ produces more noise.

The new global model parameters are:

$$
\theta_{\text{global}}=\bar{\theta}+\epsilon
$$

For example, if the average parameter is 4 and the sampled noise is 0.1:

$$
\theta_{\text{global}}=4+0.1=4.1
$$

This noise makes the global parameters less directly representative of the exact local updates.

However, Gaussian noise alone does not guarantee differential privacy. Formal differential privacy also requires controlling update sensitivity, typically through clipping, and accounting for privacy loss.

### Python Implementation

```python
def mean_models_add_noise(global_model, client_updates, noise_scale=0.1):

    global_state = global_model.state_dict()

    for item in global_state:

        updates = torch.stack(
            [c_update[item] for c_update in client_updates]
        )

        global_state[item] = (
            updates.mean(dim=0)
            + torch.normal(
                0,
                noise_scale,
                size=global_state[item].shape
            )
        )

    global_model.load_state_dict(global_state)
```

Notice the two important operations:

```python
updates.mean(dim=0)
```

This averages the parameters across clients.

```python
torch.normal(0, noise_scale, size=global_state[item].shape)
```

This generates Gaussian noise with mean zero and the specified standard deviation.

## 5. Run the Federated Learning Simulation

The next function brings everything together.

Each client:

1. Receives a copy of the global model.
2. Trains that model using its own local data.
3. Returns its trained model parameters.

The server then averages the local parameters, adds noise, and updates the global model.

The process repeats for several communication rounds.

```python
def federated_learning(num_clients, rounds, input_dim, output_dim, samples_per_client):

    global_model = SimpleModel(input_dim, output_dim)

    client_data = generate_client_data(
        num_clients,
        samples_per_client,
        input_dim
    )

    for round in range(rounds):

        print(f"Simulation {round + 1}")

        client_updates = []

        for client_id, (data, labels) in enumerate(client_data):

            local_model = SimpleModel(input_dim, output_dim)

            local_model.load_state_dict(
                global_model.state_dict()
            )

            local_update = train_local_model(
                local_model,
                data,
                labels,
                epochs=1,
                lr=0.01
            )

            client_updates.append(local_update)

        mean_models_add_noise(
            global_model,
            client_updates,
            noise_scale=0.1
        )

        test_data, ys = generate_client_data(
            1, 100, input_dim
        )[0]

        with torch.no_grad():

            preds = global_model(test_data).argmax(dim=1)

            accuracy = (
                (preds == ys).float().mean().item()
            )

        print(
            f"Global Model performance Accuracy: {accuracy:.2%}"
        )
```

Notice that the local models are initialized using the current global model:

```python
local_model.load_state_dict(global_model.state_dict())
```

This is important because each client begins its next training round from the updated global parameters.

Also notice that the server receives model parameters, not the clients' original datasets.

## 6. Run the Simulation

Finally, we define the parameters and run the simulation.

```python
num_clients = 7

simulations = 15

input_dim = 12

output_dim = 2

samples_per_each_client = 90

federated_learning(
    num_clients,
    simulations,
    input_dim,
    output_dim,
    samples_per_each_client
)
```

In this example:

- We have 7 clients.
- Each client has 90 training samples.
- Each sample contains 12 input features.
- The neural network predicts 2 classes.
- We perform 15 federated learning rounds.

## 7. Example Results

The following results are from an example execution:

```text
Simulation 1
Global Model performance Accuracy: 38.00%

Simulation 2
Global Model performance Accuracy: 40.00%

Simulation 3
Global Model performance Accuracy: 47.00%

Simulation 4
Global Model performance Accuracy: 47.00%

Simulation 5
Global Model performance Accuracy: 41.00%

Simulation 6
Global Model performance Accuracy: 50.00%

Simulation 7
Global Model performance Accuracy: 52.00%

Simulation 8
Global Model performance Accuracy: 60.00%

Simulation 9
Global Model performance Accuracy: 47.00%

Simulation 10
Global Model performance Accuracy: 46.00%

Simulation 11
Global Model performance Accuracy: 45.00%

Simulation 12
Global Model performance Accuracy: 55.00%

Simulation 13
Global Model performance Accuracy: 43.00%

Simulation 14
Global Model performance Accuracy: 38.00%

Simulation 15
Global Model performance Accuracy: 51.00%
```

The results fluctuate because the model is trained for only one local epoch per round, noise is added to the global parameters, and a new random test dataset is generated during each round.

Therefore, improvement is not guaranteed with these settings.

## 8. Federated Learning and Privacy

The important idea behind federated learning is that clients do not need to send their original training data to a central server.

Instead, they send trained model parameters.

The global model is updated using:

$$
\boxed{\theta_{\text{global}}=\frac{1}{K}\sum_{k=1}^{K}\theta_k+\epsilon}
$$

where:

$$
\epsilon\sim\mathcal{N}(0,\sigma^2)
$$

This approach has two important characteristics:

1. **Local data remains local.** Clients train using their own datasets without sending those datasets directly to the server.
2. **Noise can help obscure information.** Adding random noise reduces the direct relationship between the final global parameters and the individual local updates.

However, model updates can still leak private information. More advanced privacy-preserving federated learning systems may use secure aggregation, differential privacy, and other protection mechanisms.

### Main Takeaway

**Federated learning allows multiple clients to collaboratively train a global neural network without centralizing their original training datasets.**

Averaging local models allows the global model to learn from multiple clients, while noise addition can provide an additional layer of privacy protection.


