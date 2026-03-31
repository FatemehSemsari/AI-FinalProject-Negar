## Dataset Summary
- File: data/final/cybersecurity_intrusion_data_v1.csv
- Rows: 9537
- Columns: 11
- Target label: attack_detected

## Label Definition
- attack_detected = 0 → Normal / benign access
- attack_detected = 1 → Malicious access / attack

## Columns

| Column | Type | Description | Notes |
|------|------|-------------|-------|
| network_packet_size | int | Size of network packets in bytes | Typical range: 64–1500 |
| protocol_type | categorical | Network protocol used | TCP, UDP, ICMP |
| encryption_used | categorical | Encryption method used | AES, DES, None |
| login_attempts | int | Number of login attempts | High values may indicate brute-force |
| failed_logins | int | Number of failed login attempts | Useful for credential attacks |
| session_duration | float | Duration of the session (seconds) | Long sessions may indicate persistence |
| unusual_time_access | binary | Access at unusual time | 0 = No, 1 = Yes |
| ip_reputation_score | float | Reputation score of IP address | Higher means more suspicious |
| browser_type | categorical | Browser used | Chrome, Firefox, Edge, Safari, Unknown |
| attack_detected | binary | Target label | 0 = Normal, 1 = Attack |
