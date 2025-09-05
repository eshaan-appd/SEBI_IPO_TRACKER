import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime
import pandas as pd

st.set_page_config(page_title="SEBI RHP Tracker", layout="wide")
st.title("📑 SEBI RHP Filed with RoC — Auto Extractor")

SEBI_RHP_URL = "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=3&smid=11&ssid=15"
BASE_URL = "https://www.sebi.gov.in"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/html",
}

def get_rhp_links(pages=5):
    session = requests.Session()
    session.headers.update(HEADERS)
    all_rows = []
    for page in range(1, pages + 1):
        url = f"{SEBI_RHP_URL}&pno={page}" if page > 1 else SEBI_RHP_URL
        try:
            resp = session.get(url, timeout=30)
            soup = BeautifulSoup(resp.text, "html.parser")
            rows = soup.select("table tr")
            for r in rows:
                cols = r.find_all("td")
                if len(cols) >= 2:
                    date = cols[0].text.strip()
                    a_tag = cols[1].find("a", href=True)
                    if a_tag:
                        title = a_tag.text.strip()
                        href = urljoin(BASE_URL, a_tag["href"])
                        if "RHP" in title.upper() and not any(x in title for x in ["Addendum", "Corrigendum"]):
                            all_rows.append({"Date": date, "Title": title, "Link": href})
        except Exception as e:
            st.warning(f"Failed to load page {page}: {e}")
    return all_rows

def display_rhp_table(rhps):
    df = pd.DataFrame(rhps)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    df["RHP Release Month"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%b %Y")
    st.dataframe(df)
    st.download_button("⬇️ Download RHP List", df.to_csv(index=False), "rhp_list.csv", "text/csv")

st.sidebar.markdown("### Settings")
pages = st.sidebar.slider("Pages to scan", 1, 20, 5)

if st.sidebar.button("🚀 Run RHP Fetcher"):
    st.info("Scraping SEBI site...")
    rhps = get_rhp_links(pages)
    if rhps:
        display_rhp_table(rhps)
    else:
        st.warning("No RHP filings found. Try increasing the number of pages.")
else:
    st.info("Use the sidebar to run the tool.")
