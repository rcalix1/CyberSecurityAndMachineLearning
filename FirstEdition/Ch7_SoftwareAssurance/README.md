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


