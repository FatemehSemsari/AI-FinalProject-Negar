# Problem Statement

## Problem Definition
The goal of this project is to detect cyber intrusions by analyzing network traffic and user access behavior using supervised machine learning techniques. The task is formulated as a **binary classification problem**, where each access session is classified as either **normal** or **malicious**.

---

## Inputs
The input to the model consists of network-based and user behavior-based features extracted from each access session:

### Network-Based Inputs
- `network_packet_size`: Size of network packets in bytes
- `protocol_type`: Communication protocol (TCP, UDP, ICMP)
- `encryption_used`: Encryption method used (AES, DES, None)

### User Behavior-Based Inputs
- `login_attempts`: Number of login attempts
- `failed_logins`: Number of failed login attempts
- `session_duration`: Duration of the session in seconds
- `unusual_time_access`: Indicator of access at an unusual time (0 or 1)
- `ip_reputation_score`: Reputation score of the source IP address (0 to 1)
- `browser_type`: Browser used during the session

All categorical features are encoded, and numerical features are normalized before being fed into the model.

---

## Outputs
The output of the model is a binary label:

- `attack_detected = 1` → Malicious activity (attack)
- `attack_detected = 0` → Normal activity

Additionally, the model may produce a probability score indicating the likelihood that a given access session is malicious.

---

## Success Criteria
Due to the typically imbalanced nature of intrusion detection datasets, overall accuracy alone is not sufficient to evaluate model performance. The success of the model is measured using the following metrics:

- **F1-score**: Balances precision and recall and serves as the primary evaluation metric
- **Recall**: Emphasized to minimize false negatives, as undetected attacks pose significant security risks
- **ROC-AUC**: Measures the model’s ability to distinguish between normal and malicious access across different thresholds

A successful model achieves a high F1-score and recall while maintaining an acceptable false positive rate.

---

## Summary
This problem statement defines a supervised learning approach for intrusion detection, where meaningful network and behavioral features are used to accurately classify access events and support cybersecurity monitoring systems.
