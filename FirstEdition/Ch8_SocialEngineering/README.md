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








