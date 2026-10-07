## ML for Malware

* link

## Transfer Learning for malware detection

* Follow these links for malware data 
* https://github.com/bbdcmf/pnwcybersec
* https://github.com/JoeyShapiro/DatasetInstructions


---


# Dynamic Malware Analysis Dataset

## Overview

This dataset contains features extracted from **dynamic execution logs** of malware and goodware programs.

Dynamic malware analysis examines what a program does while it is running. Rather than examining only the executable file, we observe program behaviors such as:

- File operations
- Registry operations
- DLL usage
- System activity
- Network activity
- Process activity
- Operating system resources

These behaviors are converted into numerical feature vectors that can be used with machine learning algorithms.

---

## Dataset

The dataset is stored in:

```text
log_file_features.csv
```

The dataset contains:

```text
142 observations
1000 behavioral features
1 class label
```

The complete CSV file therefore contains:

```text
142 rows x 1001 columns
```

The machine learning feature matrix is:

```text
X = 142 samples x 1000 features
```

The class vector is:

```text
y = 142 class labels
```

---

## Classes

The final column is:

```text
class_name
```

There are two classes:

```text
class_name = -1
class_name =  1
```

The dataset contains:

```text
92 samples from one class
50 samples from the other class
```

These classes represent the malware and goodware observations used to construct the dataset.

---

## Features

Each program is represented using 1000 features extracted from its execution log.

Example feature names include:

```text
createfile
readfile
regopenkey
dll
kernel32
system32
powershell
tcp
usb
chrome
```

These features represent behaviors, operations, or resources observed while a program is executing.

For example, an execution log may contain activity associated with:

```text
CreateFile
ReadFile
RegOpenKey
kernel32.dll
System32
TCP
```

The occurrences of these terms can be counted and used as numerical features.

---

## Converting a Log into a Feature Vector

The basic process is:

```text
Program
   |
   v
Program Execution
   |
   v
Dynamic Execution Log
   |
   v
Feature Extraction
   |
   v
1000 Numerical Features
   |
   v
Machine Learning
   |
   v
Malware / Goodware Classification
```

Each program is converted into a feature vector:

```text
x = [x1, x2, x3, ..., x1000]
```

where:

```text
xi = value of behavioral feature i
```

For example:

```text
createfile = 15
readfile   = 8
regopenkey = 4
kernel32   = 21
tcp        = 2
```

Part of the resulting feature vector might therefore look like:

```text
x = [15, 8, 4, 21, 2, ...]
```

This allows a variable-length execution log to be represented by a fixed-length numerical vector.

---

## Dataset Representation

The dataset can be viewed conceptually as:

```text
             Feature 1   Feature 2   ...   Feature 1000   Class

Sample 1        x11         x12       ...      x1,1000      y1
Sample 2        x21         x22       ...      x2,1000      y2
Sample 3        x31         x32       ...      x3,1000      y3
   ...          ...         ...       ...        ...        ...
Sample 142    x142,1      x142,2      ...     x142,1000    y142
```

Therefore:

```text
X = feature data
y = class labels

X shape = (142, 1000)
y shape = (142,)
```

---

## Loading the Dataset

The dataset can be loaded using Pandas:

```python
import pandas as pd

data = pd.read_csv("log_file_features.csv")

print(data.shape)
print(data.head())
```

The features and class labels can then be separated:

```python
X = data.drop(columns=["class_name"])
y = data["class_name"]

print(X.shape)
print(y.shape)
```

Expected output:

```text
(142, 1000)
(142,)
```

---

## Machine Learning

The resulting feature vectors can be used with supervised machine learning algorithms such as:

- Neural Networks
- Logistic Regression
- Support Vector Machines
- Decision Trees
- Random Forests

The data can also be explored using unsupervised machine learning methods such as:

- K-Means
- Autoencoders
- Restricted Boltzmann Machines
- Anomaly Detection

---

## Why Dynamic Analysis?

Malware and goodware may exhibit different behaviors when they execute.

For example, malicious software may:

- Modify unusual registry locations
- Access system files
- Create or delete files
- Load particular DLLs
- Spawn processes
- Execute command-line tools
- Establish network connections

A single execution log may contain a large amount of information. Machine learning provides a way to analyze many behavioral features simultaneously.

The overall idea is:

```text
Raw program behavior
        |
        v
Numerical representation
        |
        v
Machine learning model
        |
        v
Behavior classification
```

Instead of manually examining thousands of individual log entries, the program behavior is converted into a numerical representation that can be analyzed automatically.

---


## Generating Dynamic Analysis Logs in Linux

Dynamic analysis examines the behavior of a program while the program is executing. In Linux, one simple tool for performing dynamic analysis is `strace`.

`strace` records the Linux system calls made by a running program. These calls provide information about how the program interacts with the operating system.

Examples of behavior that can be observed include:

```text
openat()     Opening files and shared libraries
read()       Reading data
write()      Writing data
execve()     Executing programs
clone()      Creating processes or threads
socket()     Creating network sockets
connect()    Making network connections
mmap()       Mapping files and libraries into memory
mprotect()   Changing memory permissions
```

Linux does not have a Windows Registry or DLL files. However, similar types of behavioral information can still be collected.

For example:

```text
Windows                    Linux

DLL activity        ->     Shared library (.so) activity
File activity       ->     openat(), read(), write()
Process activity    ->     clone(), fork(), execve()
Network activity    ->     socket(), connect(), sendto()
Memory activity     ->     mmap(), mprotect()
Registry activity   ->     No direct Linux equivalent
```

The general dynamic analysis process is:

```text
Executable Program
        |
        v
      strace
        |
        v
System Call Log
        |
        v
Feature Extraction
        |
        v
Machine Learning Dataset
```

### Install strace

On Ubuntu Linux:

```bash
sudo apt install strace
```

A simple test from the command line is:

```bash
strace -f -o execution_log.txt /bin/ls
```

Here:

```text
-f                     Trace child processes
-o execution_log.txt   Save the trace to a file
/bin/ls                Program being analyzed
```

The resulting `execution_log.txt` will contain system activity generated while `/bin/ls` executes.

For example, the log may contain entries similar to:

```text
execve("/bin/ls", ...)
openat(..., "/etc/ld.so.cache", ...)
openat(..., "libc.so.6", ...)
mmap(...)
read(...)
write(...)
close(...)
```

### Python Example

The same analysis can be performed from Python:

```python
import subprocess

program = "/bin/ls"
log_file = "execution_log.txt"

with open(log_file, "w") as log:

    subprocess.run(
        ["strace", "-f", program],
        stderr=log,
        timeout=30
    )

print("Dynamic analysis complete.")
print("Log saved to:", log_file)
```

Run the Python program:

```bash
python dynamic_analysis.py
```

The file

```text
execution_log.txt
```

* FirstEdition/Ch7_SoftwareAssurance/execution_log.txt

will then contain the system calls generated during execution of `/bin/ls`.

You can examine the log using:

```bash
cat execution_log.txt
```

or:

```bash
less execution_log.txt
```

### From Logs to Machine Learning Features

The raw `strace` log can subsequently be processed to count different types of program behavior.

For example:

```text
openat     = 34
read       = 12
write      = 7
mmap       = 21
mprotect   = 5
execve     = 1
connect    = 0
```

These values can be converted into a feature vector:

```text
x = [34, 12, 7, 21, 5, 1, 0, ...]
```

Repeating this process for many programs produces a dataset in which each row represents the dynamic behavior of one executable program.

These feature vectors can then be used for malware classification, anomaly detection, clustering, or other machine learning tasks.




---

##  Entropy score analysis of packed executable

* Paper-> https://dl.acm.org/doi/10.1145/2388576.2388607


Malware authors may pack, compress, or encrypt regions of an executable to hide malicious code and make static analysis more difficult. These packed regions often contain more random-looking byte distributions and therefore have higher Shannon entropy than surrounding structured regions. By calculating entropy across blocks of an executable, we can identify unusually high-entropy regions that may indicate packed or encrypted code.



```

import numpy as np


def entropy(data):

    counts = np.bincount(data, minlength=256)
    probabilities = counts[counts > 0] / len(data)

    return -np.sum(probabilities * np.log2(probabilities))


# ============================================================
# EXAMPLE 1: SIMULATED FILE USING HEX
#
# Imagine these are consecutive regions from an executable.
# Region 3 represents a packed/encrypted region.
# ============================================================

hex_data = """

# REGION 1 - normal
4D 5A 00 00 4D 5A 00 00 4D 5A 00 00 4D 5A 00 00

# REGION 2 - normal
10 20 10 20 10 20 10 20 10 20 10 20 10 20 10 20

# REGION 3 - packed/encrypted
A7 3C F1 82 19 DD 64 B2 EF 91 37 C8 52 0D FA 76

# REGION 4 - normal
30 40 30 40 30 40 30 40 30 40 30 40 30 40 30 40

"""


# Remove comments and convert hexadecimal to integers

values = []

for line in hex_data.splitlines():

    line = line.strip()

    if line and not line.startswith("#"):

        values.extend(
            int(x, 16) for x in line.split()
        )


exe = np.array(values, dtype=np.uint8)


# Each region contains 16 bytes

block_size = 16


print("HEX EXAMPLE")

for i in range(0, len(exe), block_size):

    block = exe[i:i + block_size]

    H = entropy(block)

    print(
        "Region", i // block_size + 1,
        "Entropy:", round(H, 3)
    )


# ============================================================
# EXAMPLE 2: SAME IDEA USING RANDOM DATA
#
# Simulate a larger executable with a packed region
# in the middle.
# ============================================================

normal1 = np.random.randint(0, 40, 2000, dtype=np.uint8)

packed = np.random.randint(0, 256, 1000, dtype=np.uint8)

normal2 = np.random.randint(0, 40, 2000, dtype=np.uint8)


exe = np.concatenate((normal1, packed, normal2))


block_size = 500


print("\nRANDOM EXAMPLE")

for i in range(0, len(exe), block_size):

    block = exe[i:i + block_size]

    H = entropy(block)

    print(
        "Bytes", i, "-", i + len(block) - 1,
        "Entropy:", round(H, 3)
    )

```

For an EXE

```

import numpy as np


def entropy(data):

    counts = np.bincount(data, minlength=256)
    probabilities = counts[counts > 0] / len(data)

    return -np.sum(probabilities * np.log2(probabilities))


# ============================================================
# EXAMPLE 3: READ A REAL EXECUTABLE FILE
# ============================================================

filename = "program.exe"


# Read the executable as raw bytes

with open(filename, "rb") as f:

    raw_data = f.read()


# Convert raw bytes to integers from 0 to 255

exe = np.frombuffer(raw_data, dtype=np.uint8)


# Analyze the executable in blocks

block_size = 1024


for i in range(0, len(exe), block_size):

    block = exe[i:i + block_size]

    H = entropy(block)

    print(
        "Bytes", i, "-", i + len(block) - 1,
        "Entropy:", round(H, 3)
    )

```

---


## Synthetic system-call sequences, learned embeddings, and LSTM

* Paper: https://www.astesj.com/v05/i04/p26/?


```


import torch
import random
from torch import nn


# ============================================================
# SYSTEM CALLS
# ============================================================

calls = [
    "openat",
    "read",
    "write",
    "close",
    "mmap",
    "mprotect",
    "execve",
    "socket",
    "connect"
]

call_to_id = {call: i for i, call in enumerate(calls)}


# ============================================================
# CREATE RANDOM SYSTEM-CALL SEQUENCES
#
# 0 = normal
# 1 = suspicious
# ============================================================

normal_calls = [
    "openat",
    "read",
    "write",
    "close",
    "mmap"
]

suspicious_calls = [
    "mprotect",
    "execve",
    "socket",
    "connect"
]


X = []
y = []


# normal sequences

for i in range(100):

    sequence = random.choices(normal_calls, k=10)

    X.append(sequence)

    y.append(0)


# suspicious sequences

for i in range(100):

    sequence = random.choices(calls, k=6)

    sequence += random.choices(suspicious_calls, k=4)

    random.shuffle(sequence)

    X.append(sequence)

    y.append(1)


# ============================================================
# SYSTEM CALLS -> INTEGER IDs
# ============================================================

X = torch.tensor([
    [call_to_id[call] for call in sequence]
    for sequence in X
])

y = torch.tensor(y)


print("Example Sequence:")
print(X[0])


# ============================================================
# LSTM
# ============================================================

class Net(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            len(calls),
            4
        )

        self.lstm = nn.LSTM(
            input_size=4,
            hidden_size=8,
            batch_first=True
        )

        self.fc = nn.Linear(8, 2)


    def forward(self, x):

        # system call IDs -> learned vectors

        x = self.embedding(x)

        output, (hidden, cell) = self.lstm(x)

        x = hidden[-1]

        return self.fc(x)


model = Net()


# ============================================================
# TRAIN
# ============================================================

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


for epoch in range(100):

    prediction = model(X)

    loss = loss_fn(prediction, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    if epoch % 20 == 0:

        print(
            "Epoch:",
            epoch,
            "Loss:",
            round(loss.item(), 4)
        )


# ============================================================
# SHOW LEARNED EMBEDDINGS
# ============================================================

print("\nLearned Embeddings:")

for call in calls:

    call_id = torch.tensor(call_to_id[call])

    vector = model.embedding(call_id)

    print(
        call,
        "->",
        vector.detach().numpy()
    )


# ============================================================
# TEST
# ============================================================

test = [
    "openat",
    "read",
    "mprotect",
    "execve",
    "socket",
    "connect",
    "read",
    "write",
    "close",
    "close"
]


test = torch.tensor([
    [call_to_id[call] for call in test]
])


with torch.no_grad():

    prediction = model(test)

    predicted_class = prediction.argmax(dim=1)


print("\nPrediction:", predicted_class.item())


# 0 = normal
# 1 = suspicious





```



Read form STRACE log

```


import re


# ============================================================
# READ STRACE FILE
# ============================================================

filename = "execution_log.txt"

system_calls = []


with open(filename, "r") as f:

    for line in f:

        # extract system call name

        match = re.match(r"([a-zA-Z0-9_]+)\(", line)

        if match:

            system_calls.append(match.group(1))


# ============================================================
# SHOW SYSTEM CALLS
# ============================================================

print("System Calls:")

print(system_calls)


# ============================================================
# BREAK INTO SEQUENCES OF 10
# ============================================================

sequence_length = 10

X = []


for i in range(0, len(system_calls) - sequence_length + 1, sequence_length):

    sequence = system_calls[i:i + sequence_length]

    X.append(sequence)


# ============================================================
# DISPLAY SEQUENCES
# ============================================================

print("\nSequences:")

for sequence in X:

    print(sequence)


# ============================================================
# ADD LABEL
#
# This trace came from a known normal execution.
# ============================================================

y = [0] * len(X)


print("\nLabels:")

print(y)


```






---


