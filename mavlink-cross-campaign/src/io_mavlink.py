"""Parse MAVLink message-ID sequences out of pcap/pcapng captures.

Each UDP payload may carry several MAVLink frames back to back:
  v1: 0xFE LEN SEQ SYS COMP MSGID(1)            PAYLOAD CRC(2)        -> 8 + LEN bytes
  v2: 0xFD LEN INC CMP SEQ SYS COMP MSGID(3,LE) PAYLOAD CRC(2) [SIG13] -> 12 + LEN (+13 if INC & 1)
One output row per MAVLink frame.
"""
from __future__ import annotations

import socket

import dpkt
import numpy as np
import pandas as pd


def _reader(fh):
    head = fh.read(4); fh.seek(0)
    return dpkt.pcapng.Reader(fh) if head == b"\x0a\x0d\x0d\x0a" else dpkt.pcap.Reader(fh)


def _l2_to_ip(buf, linktype):
    if linktype == dpkt.pcap.DLT_EN10MB:
        eth = dpkt.ethernet.Ethernet(buf)
        return eth.data, eth.src, eth.dst
    if linktype in (dpkt.pcap.DLT_RAW, 101):
        return dpkt.ip.IP(buf), b"", b""
    if linktype == dpkt.pcap.DLT_LINUX_SLL:
        sll = dpkt.sll.SLL(buf)
        return sll.data, sll.hdr[:6] if hasattr(sll, "hdr") else b"", b""
    if linktype == dpkt.pcap.DLT_NULL:
        return dpkt.loopback.Loopback(buf).data, b"", b""
    raise ValueError(f"unsupported linktype {linktype}")


def split_mavlink(payload: bytes):
    """Yield (version, seq, sysid, compid, msgid, length) for each frame in a UDP payload."""
    i, n = 0, len(payload)
    while i < n:
        stx = payload[i]
        if stx == 0xFE and i + 8 <= n:
            ln = payload[i + 1]
            yield 1, payload[i + 2], payload[i + 3], payload[i + 4], payload[i + 5], ln
            i += 8 + ln
        elif stx == 0xFD and i + 12 <= n:
            ln, inc = payload[i + 1], payload[i + 2]
            msgid = payload[i + 7] | (payload[i + 8] << 8) | (payload[i + 9] << 16)
            yield 2, payload[i + 4], payload[i + 5], payload[i + 6], msgid, ln
            i += 12 + ln + (13 if inc & 1 else 0)
        else:
            return  # not MAVLink (or truncated); stop at the first unparsable byte


def mac(b: bytes) -> str:
    return ":".join(f"{x:02x}" for x in b) if b else ""


def load_pcap(path) -> pd.DataFrame:
    """Columns: t (s since first packet), src, dst (ip:port), src_mac, ver, seq, sysid,
    compid, msgid, mlen, pkt (index of the UDP datagram the frame came from)."""
    rows = []
    with open(path, "rb") as fh:
        rd = _reader(fh)
        lt = rd.datalink()
        for k, (ts, buf) in enumerate(rd):
            try:
                ip, smac, _ = _l2_to_ip(buf, lt)
            except Exception:
                continue
            if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)) or not isinstance(ip.data, dpkt.udp.UDP):
                continue
            udp = ip.data
            fam = socket.AF_INET if isinstance(ip, dpkt.ip.IP) else socket.AF_INET6
            src = f"{socket.inet_ntop(fam, ip.src)}:{udp.sport}"
            dst = f"{socket.inet_ntop(fam, ip.dst)}:{udp.dport}"
            for ver, seq, sysid, compid, msgid, ln in split_mavlink(bytes(udp.data)):
                rows.append((ts, src, dst, mac(smac), ver, seq, sysid, compid, msgid, ln, k))
    df = pd.DataFrame(rows, columns=["t", "src", "dst", "src_mac", "ver", "seq", "sysid",
                                     "compid", "msgid", "mlen", "pkt"])
    if len(df):
        df["t"] -= df["t"].iloc[0]
    for c in ("ver", "seq", "sysid", "compid", "mlen"):
        df[c] = df[c].astype(np.int16)
    df["msgid"] = df["msgid"].astype(np.int32)
    return df
