import numpy as np
import pandas as pd

def add_security_featchers(df):
    df["login_pressure"] = df["login_attempts"] * df["failed_logins"]
    df["time_fail_risk"] = df["unusual_time_access"] * df["failed_logins"]
    df["traffic_intensity"] = df["network_packet_size"] * df["session_duration"]
    df["ip_attack_risk"] = df["ip_reputation_score"] * df["failed_logins"]
    return df