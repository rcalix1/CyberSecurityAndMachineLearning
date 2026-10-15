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
