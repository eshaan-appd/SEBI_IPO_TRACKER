import os
import re
import json
import time
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin

import requests
import pandas as pd
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import streamlit as st

# ============================
# Constants
# ============================

SEBI_PUBLIC_ISSUES_RHP = "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=3&smid=11&ssid=15"
SEBI_PUBLIC_ISSUES_DRHP = "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=3&smid=10&ssid=15"
SEBI_HOME = "https://www.sebi.gov.in/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Referer": SEBI_HOME,
}

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_listing_page(_session: requests.Session, page_no: int) -> Tuple[str, List[Dict]]:
    url = f"{SEBI_PUBLIC_ISSUES_RHP}&pno={page_no}" if page_no > 1 else SEBI_PUBLIC_ISSUES_RHP
    resp = _session.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    items = []
    for tr in soup.select("table tr"):
        tds = tr.find_all("td")
        if len(tds) >= 2:
            date_str = tds[0].text.strip()
            link = tds[1].find("a", href=True)
            if link:
                items.append({
                    "date_str": date_str,
                    "title": link.text.strip(),
                    "url": urljoin(url, link["href"])
                })
    return url, items

@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_detail_page(_session: requests.Session, url: str) -> Tuple[str, List[str]]:
    resp = _session.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    pdfs = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True) if a["href"].lower().endswith(".pdf")]
    return resp.text, pdfs

st.title("📄 SEBI RHP Extractor")
st.markdown("✅ Streamlit session handling error resolved using `_session` instead of `session` in cached functions.")
