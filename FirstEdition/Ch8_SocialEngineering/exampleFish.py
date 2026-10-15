

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


