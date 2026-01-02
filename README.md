## About Dataset

This project uses a **Cybersecurity Intrusion Detection Dataset** designed to identify malicious activities based on **network traffic characteristics** and **user access behavior**. Each record in the dataset represents a single network session or access event and captures both technical and behavioral aspects of API or system usage.

The dataset combines low-level network information with high-level user behavior features, enabling effective detection of cyber intrusions such as brute-force attacks, unauthorized access, credential misuse, and abnormal usage patterns.

### Dataset Features

The features are grouped into two main categories:

#### Network-Based Features
These features describe the characteristics of network communication during a session:
- **network_packet_size**: Size of network packets, which may indicate abnormal traffic patterns
- **protocol_type**: Communication protocol used (TCP, UDP, ICMP)
- **encryption_used**: Encryption method applied (AES, DES, or None)

#### User Behavior-Based Features
These features capture user activity and access behavior:
- **login_attempts**: Number of login attempts during a session
- **failed_logins**: Number of unsuccessful login attempts
- **session_duration**: Length of the session in seconds
- **unusual_time_access**: Indicates access during unusual hours
- **ip_reputation_score**: Reputation score of the IP address (0 to 1)
- **browser_type**: Browser used for access (Chrome, Firefox, Edge, Safari, or Unknown)

### Target Variable
- **attack_detected**:
  - `1` indicates malicious activity (attack)
  - `0` indicates normal behavior

The presence of labeled attack data makes this dataset well-suited for **supervised machine learning**, where models learn to distinguish between normal and malicious access patterns.

---

## Dataset Objectives

- Provide realistic network traffic and user behavior data for intrusion detection
- Enable supervised classification of normal and malicious access events
- Support the development of machine learning-based intrusion detection systems (IDS)
- Allow evaluation of security models using standard classification metrics
- Reflect real-world cybersecurity challenges such as evolving attack patterns and class imbalance
