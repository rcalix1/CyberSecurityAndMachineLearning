## ML for Phishing

* link

---


## Phishing Website Detection Dataset

This dataset is used to demonstrate the application of machine learning algorithms to phishing website detection.

The dataset (`phishing.csv`) contains **11,055 website samples**, each represented by **30 features** describing characteristics of a website and its URL.

The features include:

- **URL characteristics:** URL length, IP address usage, URL shortening services, and special symbols.
- **Domain information:** Domain age, registration length, DNS records, and subdomains.
- **Website security:** SSL certificate status, HTTPS tokens, and abnormal URLs.
- **HTML and JavaScript behavior:** Redirects, pop-up windows, iframes, and mouseover events.
- **Website reputation:** Web traffic, page rank, Google indexing, and statistical reports.

### Classification Labels

The dataset includes a target column named `Result`:

- `-1`: Phishing website (4,898 samples)
- `1`: Legitimate website (6,157 samples)

Most input features are encoded using values of `-1`, `0`, and `1`, representing different feature conditions.

### Machine Learning Application

The objective is to train a supervised machine learning classifier to distinguish phishing websites from legitimate websites based on their extracted features.

The dataset can be used with algorithms such as Support Vector Machines (SVMs), Decision Trees, Random Forests, and Neural Networks.

Students can evaluate classifier performance using accuracy, precision, recall, F1-score, and confusion matrices.


## Feature Extraction



## Phishing Email: Data Collection and Feature Extraction

This example demonstrates how phishing-related data can be collected from emails and converted into numerical features for machine learning.

### Methodology

1. Create a sample phishing email (`.eml` file).
2. Read the email using Python's `email` library.
3. Extract URLs from the email body using regular expressions.
4. Extract numerical features from each URL.
5. Save the extracted features into a CSV file for machine learning.

### Extracted Features

| Feature | Description |
|---|---|
| URL_Length | Number of characters in the URL |
| Having_IP_Address | Whether the URL hostname is an IP address |
| Having_At_Symbol | Whether the URL contains an @ symbol |
| Number_of_Dots | Number of dots in the hostname |
| HTTPS | Whether the URL uses HTTPS |

### Python Libraries

- `email`: Reading and processing email files.
- `re`: Extracting URLs using regular expressions.
- `urllib.parse`: Parsing URL components.
- `ipaddress`: Identifying IP addresses.
- `pandas`: Creating and saving the dataset.

### Machine Learning Application

Each extracted URL is represented by a numerical feature vector:

X = [URL_Length, Having_IP_Address, Having_At_Symbol, Number_of_Dots, HTTPS]

These features can be used to train supervised machine learning algorithms such as Support Vector Machines (SVM), Random Forests, and Neural Networks.

For supervised learning, each sample must also have a verified label indicating whether it is phishing or legitimate.

**Note:** This example demonstrates feature extraction only. It does not determine whether a website is malicious. HTTPS alone does not guarantee that a website is legitimate.



```


import email
import re
import pandas as pd
from email.message import EmailMessage
from urllib.parse import urlparse
import ipaddress

########################################
# 1. Create a sample phishing email
########################################

msg = EmailMessage()

msg["From"] = "security@example.com"
msg["To"] = "student@example.com"
msg["Subject"] = "Urgent: Verify Your Account"

msg.set_content("""
Dear Customer,

Your account has been temporarily suspended.

Please verify your account immediately:

http://192.0.2.10/login/verify-account

You may also visit:

https://secure-account.example.com/login

Thank you,
Security Team
""")

with open("phishing_email.eml", "wb") as f:
    f.write(msg.as_bytes())

########################################
# 2. Read the email
########################################

with open("phishing_email.eml", "rb") as f:
    message = email.message_from_binary_file(f)

body = message.get_payload(decode=True).decode(
    errors="ignore"
)

print("Email Subject:", message["Subject"])
print("Email Body:", body)

########################################
# 3. Extract URLs
########################################

urls = re.findall(r'https?://[^\s<>"\']+', body)

########################################
# 4. Extract numerical features
########################################

data = []

for url in urls:

    parsed = urlparse(url)
    domain = parsed.hostname or ""

    try:
        ipaddress.ip_address(domain)
        has_ip = 1
    except ValueError:
        has_ip = 0

    features = {
        "URL": url,
        "URL_Length": len(url),
        "Having_IP_Address": has_ip,
        "Having_At_Symbol": int("@" in url),
        "Number_of_Dots": domain.count("."),
        "HTTPS": int(parsed.scheme == "https")
    }

    data.append(features)

########################################
# 5. Create dataset
########################################

df = pd.DataFrame(data)

print("\nExtracted Features:")
print(df.to_string(index=False))

df.to_csv("email_url_features.csv", index=False)


```




---


## Phishing Campaign Graph Analysis

This example demonstrates how graph theory can be used to investigate phishing campaigns by identifying relationships between emails, domains, and IP addresses.

Rather than classifying individual emails, the objective is to discover groups of potentially related phishing activities based on shared network infrastructure.

### Methodology

1. Generate six simulated phishing emails in `.eml` format.
2. Read the emails using Python's `email` library.
3. Extract URLs and domain names using regular expressions.
4. Associate each domain with an IP address using simulated DNS records.
5. Construct a graph using NetworkX.
6. Identify connected components representing potential phishing campaigns.
7. Visualize the graph using Matplotlib.

### Graph Representation

A graph is defined as:

$$
G = (V,E)
$$

where:

- **V** represents nodes consisting of emails, domains, and IP addresses.
- **E** represents relationships between nodes.

The graph contains three types of nodes:

| Node Type | Description |
|---|---|
| Email | An email containing a URL |
| Domain | Domain extracted from the URL |
| IP Address | IP address associated with the domain |

Edges connect emails to domains and domains to IP addresses.

### Connected Components

A connected component is a set of nodes in which every node can be reached from every other node through a sequence of edges.

The example generates three connected components:

- Campaign 1: Three emails sharing an IP address.
- Campaign 2: Two emails sharing another IP address.
- Campaign 3: One isolated email and its associated infrastructure.

### Python Libraries

- `email`: Creating and reading email files.
- `re`: Extracting URLs from email content.
- `urllib.parse`: Extracting domain names.
- `networkx`: Constructing graphs and identifying connected components.
- `matplotlib`: Visualizing the network.

### Cybersecurity Application

Graph analysis can help security analysts identify potentially coordinated phishing campaigns, shared infrastructure, and relationships between suspicious websites.

This methodology can be extended using graph clustering, community detection, and Graph Neural Networks (GNNs).

**Note:** The example uses simulated emails and DNS records. Shared IP addresses suggest possible relationships but do not necessarily establish that websites belong to the same attacker.



```


import networkx as nx
import matplotlib.pyplot as plt
from email.message import EmailMessage
from email import policy
from email.parser import BytesParser
from urllib.parse import urlparse
import re

########################################
# 1. Create sample phishing emails
########################################

records = [
    ("email1", "http://login-alert.example"),
    ("email2", "http://verify-account.example"),
    ("email3", "http://security-update.example"),
    ("email4", "http://payment-check.example"),
    ("email5", "http://billing-alert.example"),
    ("email6", "http://unrelated.example")
]

# Simulated DNS records
# Real applications would query DNS
dns = {
    "login-alert.example": "192.0.2.10",
    "verify-account.example": "192.0.2.10",
    "security-update.example": "192.0.2.10",
    "payment-check.example": "198.51.100.20",
    "billing-alert.example": "198.51.100.20",
    "unrelated.example": "203.0.113.30"
}

for name, url in records:

    msg = EmailMessage()
    msg["From"] = "support@example.com"
    msg["To"] = "student@example.com"
    msg["Subject"] = "Verify Your Account"
    msg.set_content(f"Please verify your account: {url}")

    with open(name + ".eml", "wb") as f:
        f.write(msg.as_bytes())

########################################
# 2. Read emails and construct graph
########################################

G = nx.Graph()

for name, _ in records:

    with open(name + ".eml", "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)

    body = msg.get_body(preferencelist=("plain",))
    text = body.get_content()

    urls = re.findall(r"https?://[^\s<>\"']+", text)

    for url in urls:

        domain = urlparse(url).hostname
        ip = dns[domain]

        G.add_node(name, type="email")
        G.add_node(domain, type="domain")
        G.add_node(ip, type="ip")

        G.add_edge(name, domain)
        G.add_edge(domain, ip)

########################################
# 3. Detect connected components
########################################

clusters = list(nx.connected_components(G))

print("Potential phishing campaigns:", len(clusters))

for i, cluster in enumerate(clusters):
    print(f"\nCampaign {i+1}")
    print(sorted(cluster))

########################################
# 4. Visualize graph
########################################

colors = {
    "email": "lightblue",
    "domain": "orange",
    "ip": "lightgreen"
}

node_colors = [
    colors[G.nodes[n]["type"]]
    for n in G.nodes()
]

plt.figure(figsize=(13, 8))

nx.draw(
    G,
    pos=nx.spring_layout(G, seed=42),
    with_labels=True,
    node_color=node_colors,
    node_size=2200,
    font_size=8,
    edge_color="gray"
)

plt.title("Phishing Campaign Graph Analysis")
plt.tight_layout()
plt.show()




```



---

## Node to Vec




## Phishing Campaign Detection with Node2Vec and K-Means

This example demonstrates how graph representation learning and unsupervised machine learning can be used to discover potentially related phishing campaigns.

Node2Vec is used to learn numerical embeddings of graph nodes representing emails, domains, and IP addresses. K-Means clustering is then applied to identify groups of emails with similar network relationships.

### Methodology

1. Generate simulated phishing infrastructure containing emails, domains, and IP addresses.
2. Construct an undirected graph using NetworkX.
3. Apply Node2Vec to generate random walks through the graph.
4. Train a Word2Vec-based model to learn numerical node embeddings.
5. Extract the learned embeddings for each email.
6. Apply K-Means clustering to group emails based on their embeddings.
7. Use Principal Component Analysis (PCA) to visualize the clusters in two dimensions.

### Graph Representation

The graph is defined as:

$$
G = (V,E)
$$

where:

- **V** represents emails, domains, and IP addresses.
- **E** represents relationships between these entities.

### Node2Vec

Node2Vec learns a mapping from graph nodes to numerical vectors:

$$
f: V \rightarrow \mathbb{R}^{d}
$$

where $d$ is the embedding dimension.

The algorithm generates biased random walks through the graph and learns embeddings by maximizing the probability of observing neighboring nodes within the sampled walks.

A simplified objective is:

$$
\max_f \sum_{u \in V}
\sum_{v \in N(u)}
\log P(v \mid f(u))
$$

where $N(u)$ represents the sampled context nodes of vertex $u$.

```

import random
import torch
from torch import nn

########################################
# 1. Graph
########################################

graph = {
    0: [1],
    1: [0, 2],
    2: [1, 3],
    3: [2, 4],
    4: [3],
    5: [6],
    6: [5, 7],
    7: [6]
}

########################################
# 2. Random walks
########################################

walks = []

for node in graph:
    for _ in range(100):
        walk = [node]

        for _ in range(5):
            neighbors = graph[walk[-1]]
            walk.append(random.choice(neighbors))

        walks.append(walk)

########################################
# 3. Training pairs (Skip-gram)
########################################

X, Y = [], []

for walk in walks:
    for i in range(len(walk)):
        for j in range(max(0, i-2),
                       min(len(walk), i+3)):

            if i != j:
                X.append(walk[i])
                Y.append(walk[j])

X = torch.tensor(X)
Y = torch.tensor(Y)

########################################
# 4. Neural network
########################################

class Node2Vec(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(8, 2)
        self.output = nn.Linear(2, 8)

    def forward(self, x):
        h = self.embedding(x)
        return self.output(h)

model = Node2Vec()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

########################################
# 5. Train
########################################

for epoch in range(200):

    optimizer.zero_grad()

    pred = model(X)
    loss = loss_fn(pred, Y)

    loss.backward()
    optimizer.step()

########################################
# 6. Learned embeddings
########################################

print(model.embedding.weight.data)


```


## Phishing Campaign Analysis with Node2Vec (From Scratch)

This example demonstrates graph representation learning using PyTorch. The objective is to learn numerical embeddings of emails, domains, and IP addresses based on their relationships in a phishing infrastructure graph.

The implementation uses random walks and a Skip-gram neural network, similar to Word2Vec.

### 1. Graph Construction

The graph consists of three types of nodes:

- Emails
- Domains
- IP addresses

For example:

```text
Email1 -- DomainA -- IP1 -- DomainB -- Email2

Email3 -- DomainC -- IP2 -- DomainD -- Email4
```

Emails 1 and 2 are connected through shared infrastructure, while Emails 3 and 4 form another connected group.

The graph is represented in Python using a dictionary containing each node and its neighbors.

### 2. Random Walks

Random walks are generated by repeatedly selecting a neighboring node.

For example:

```text
Email1 → DomainA → IP1 → DomainB → Email2
Email2 → DomainB → IP1 → DomainA → Email1
Email3 → DomainC → IP2 → DomainD → Email4
```

These sequences are analogous to sentences in natural language processing.

In Word2Vec, neighboring words provide training examples. In graph representation learning, neighboring nodes in random walks provide training examples.

### 3. Skip-gram Neural Network

Each node is assigned an integer identifier, which is equivalent to selecting its one-hot encoded representation.

The neural network consists of:

- Input: Node identifier.
- Embedding layer: Two neurons representing the learned node embedding.
- Output layer: Predicts neighboring nodes.

The embedding is computed as:

$$
\mathbf{h}=W^\top\mathbf{x}
$$

where $\mathbf{x}$ is the one-hot encoded node and $\mathbf{h}$ is its learned embedding.

The output probabilities are:

$$
\hat{\mathbf{y}}=\operatorname{softmax}(V\mathbf{h}+\mathbf{b})
$$

### 4. Training

Training pairs are generated using a context window of two positions on either side of each node in the random walks.

The network is trained using cross-entropy loss:

$$
\mathcal{L}=-\log P(v_{\text{context}}\mid v_{\text{input}})
$$

The objective is to learn embeddings that predict nodes appearing nearby in the random walks.

### 5. Learned Embeddings

After training, each node is represented by a two-dimensional vector:

$$
f:V\rightarrow\mathbb{R}^{2}
$$

Nodes appearing in similar graph contexts tend to develop similar embeddings.

The learned embeddings are visualized using a two-dimensional scatter plot.

### 6. Cybersecurity Application

Graph embeddings can be used to investigate relationships among phishing emails, suspicious domains, and shared network infrastructure.

The learned vectors can also serve as inputs to machine learning algorithms such as K-Means, SVMs, or neural networks.

Unlike traditional phishing classifiers, this approach learns from the structure of relationships between entities rather than only from individual email or website features.

**Note:** This educational implementation uses uniform random walks, making it a simplified DeepWalk-style model. Full Node2Vec introduces biased random walks controlled by parameters $p$ and $q$.

### References

1. Perozzi, B., Al-Rfou, R., and Skiena, S. (2014). "DeepWalk: Online Learning of Social Representations." Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 701–710.

2. Grover, A., and Leskovec, J. (2016). "node2vec: Scalable Feature Learning for Networks." Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 855–864.




```

import random
import torch
from torch import nn

########################################
# 1. Phishing graph
########################################

graph = {
    "Email1":  ["DomainA"],
    "DomainA": ["Email1", "IP1"],
    "IP1":     ["DomainA", "DomainB"],
    "DomainB": ["IP1", "Email2"],
    "Email2":  ["DomainB"],

    "Email3":  ["DomainC"],
    "DomainC": ["Email3", "IP2"],
    "IP2":     ["DomainC", "DomainD"],
    "DomainD": ["IP2", "Email4"],
    "Email4":  ["DomainD"]
}

########################################
# 2. Random walks (sentences)
########################################

random.seed(42)
torch.manual_seed(42)

walks = []

for node in graph:
    for _ in range(100):
        walk = [node]

        for _ in range(5):
            neighbors = graph[walk[-1]]
            walk.append(random.choice(neighbors))

        walks.append(walk)

print("Example random walks:")
for walk in walks[:5]:
    print(walk)

########################################
# 3. Convert node names to integers
########################################

nodes = list(graph.keys())

node_to_id = {
    node: i for i, node in enumerate(nodes)
}

########################################
# 4. Training pairs (Skip-gram)
########################################

X, Y = [], []

for walk in walks:
    for i in range(len(walk)):
        for j in range(max(0, i-2),
                       min(len(walk), i+3)):

            if i != j:
                X.append(node_to_id[walk[i]])
                Y.append(node_to_id[walk[j]])

X = torch.tensor(X)
Y = torch.tensor(Y)

########################################
# 5. Neural network
########################################

class Node2Vec(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(len(nodes), 2)
        self.output = nn.Linear(2, len(nodes))

    def forward(self, x):
        h = self.embedding(x)
        return self.output(h)

model = Node2Vec()

optimizer = torch.optim.Adam(
    model.parameters(), lr=0.01
)

loss_fn = nn.CrossEntropyLoss()

########################################
# 6. Train
########################################

for epoch in range(200):

    optimizer.zero_grad()

    pred = model(X)
    loss = loss_fn(pred, Y)

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print("Epoch:", epoch,
              "Loss:", loss.item())

########################################
# 7. Learned node embeddings
########################################

print("\nLearned Embeddings:")

with torch.no_grad():
    embeddings = model.embedding.weight

    for node, vector in zip(nodes, embeddings):
        print(node, vector.numpy().round(3))

########################################
# 8. Visualize embeddings
########################################

import matplotlib.pyplot as plt

E = model.embedding.weight.detach().numpy()

plt.figure(figsize=(8, 6))

plt.scatter(E[:, 0], E[:, 1], s=100)

for i, node in enumerate(nodes):
    plt.annotate(
        node,
        (E[i, 0], E[i, 1]),
        xytext=(5, 5),
        textcoords="offset points"
    )

plt.xlabel("Embedding 1")
plt.ylabel("Embedding 2")
plt.title("Phishing Graph Node Embeddings")
plt.grid(True)
plt.show()


```


### K-Means Clustering

The learned email embeddings are grouped using K-Means:

$$
\min_{\mu_1,\ldots,\mu_K}
\sum_{k=1}^{K}
\sum_{\mathbf{x}_i \in C_k}
\|\mathbf{x}_i-\mu_k\|^2
$$

where:

- $K$ is the number of clusters.
- $C_k$ is the set of embeddings assigned to cluster $k$.
- $\mu_k$ is the centroid of cluster $k$.

### Visualization

Principal Component Analysis (PCA) reduces the eight-dimensional node embeddings to two dimensions for visualization.

Each point represents an email, and its color indicates the cluster assigned by K-Means.

### Python Libraries

- `networkx`: Graph construction.
- `node2vec`: Graph embedding learning.
- `gensim`: Word2Vec training.
- `scikit-learn`: K-Means clustering and PCA.
- `matplotlib`: Visualization.
- `numpy`: Numerical operations.

### Cybersecurity Application

This methodology can help identify phishing emails associated with shared infrastructure, discover relationships between suspicious domains, and investigate potentially coordinated phishing campaigns.

Unlike connected components detection, Node2Vec uses machine learning to generate numerical representations of graph structure.

**Note:** The example uses simulated infrastructure and a small dataset. Three clusters are specified in advance for demonstration purposes. The resulting clusters do not necessarily establish that emails belong to the same phishing campaign.

### Reference

Grover, A., and Leskovec, J. (2016). "node2vec: Scalable Feature Learning for Networks." Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 855–864.




```


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from node2vec import Node2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

########################################
# 1. Simulated phishing infrastructure
########################################

records = [
    ("email1", "login-alert.example", "192.0.2.10"),
    ("email2", "verify-account.example", "192.0.2.10"),
    ("email3", "security-update.example", "192.0.2.10"),
    ("email4", "payment-check.example", "198.51.100.20"),
    ("email5", "billing-alert.example", "198.51.100.20"),
    ("email6", "unrelated.example", "203.0.113.30")
]

G = nx.Graph()

for email, domain, ip in records:
    G.add_edge(email, domain)
    G.add_edge(domain, ip)

########################################
# 2. Node2Vec representation learning
########################################

node2vec = Node2Vec(
    G,
    dimensions=8,
    walk_length=10,
    num_walks=100,
    workers=1,
    seed=42
)

model = node2vec.fit(
    window=5,
    min_count=1,
    batch_words=4,
    seed=42
)

########################################
# 3. Extract email embeddings
########################################

emails = [r[0] for r in records]

X = np.array([
    model.wv[email] for email in emails
])

print("Embedding matrix:", X.shape)
print(X)

########################################
# 4. K-Means clustering
########################################

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(X)

for email, label in zip(emails, labels):
    print(email, "Cluster:", label)

########################################
# 5. PCA visualization
########################################

pca = PCA(n_components=2)
X2 = pca.fit_transform(X)

plt.figure(figsize=(8, 6))

plt.scatter(
    X2[:, 0],
    X2[:, 1],
    c=labels,
    cmap="viridis",
    s=180
)

for i, email in enumerate(emails):
    plt.annotate(
        email,
        (X2[i, 0], X2[i, 1]),
        xytext=(5, 5),
        textcoords="offset points"
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Node2Vec Phishing Campaign Clustering")
plt.tight_layout()
plt.show()


```










---

