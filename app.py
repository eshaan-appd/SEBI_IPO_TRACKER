
import os
import re
import json
import time
import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime
from openai import OpenAI

# Config
SEBI_RHP_URL = "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=3&smid=11&ssid=15"
BASE_URL = "https://www.sebi.gov.in"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Streamlit UI
st.set_page_config(page_title="SEBI RHP Extractor", layout="wide")
st.title("📄 SEBI RHP Filed with RoC — Auto Extractor via OpenAI")

st.sidebar.header("Settings")
openai_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")
model = st.sidebar.selectbox("Model", ["gpt-4-1106-preview", "gpt-4o"], index=1)
pages = st.sidebar.slider("Pages to scan", 1, 10, 3)
start_btn = st.sidebar.button("🚀 Run Extraction")

@st.cache_data(show_spinner=False)
def get_rhp_list(pages):
    session = requests.Session()
    session.headers.update(HEADERS)
    data = []
    for p in range(1, pages + 1):
        url = f"{SEBI_RHP_URL}&pno={p}" if p > 1 else SEBI_RHP_URL
        try:
            resp = session.get(url, timeout=20)
            soup = BeautifulSoup(resp.text, "html.parser")
            for row in soup.select("table tr"):
                cols = row.find_all("td")
                if len(cols) >= 2:
                    date = cols[0].text.strip()
                    a = cols[1].find("a", href=True)
                    if a and "RHP" in a.text.upper() and not any(x in a.text for x in ["Addendum", "Corrigendum"]):
                        link = urljoin(BASE_URL, a["href"])
                        data.append({"date": date, "title": a.text.strip(), "url": link})
        except Exception as e:
            st.warning(f"Failed to load page {p}: {e}")
    return data

def get_pdf_link(detail_url):
    try:
        resp = requests.get(detail_url, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.lower().endswith(".pdf"):
                return urljoin(detail_url, href)
        for iframe in soup.find_all("iframe", src=True):
            src = iframe["src"].strip()
            if src.lower().endswith(".pdf"):
                return urljoin(detail_url, src)
    except Exception as e:
        print(f"Error in get_pdf_link: {e}")
        return None
    return None

def build_prompt(title, date_hint):
    return f'''
You are an expert in IPO prospectus analysis. From the attached Indian IPO RHP PDF, extract the following fields:

- company_name
- sector (e.g., Healthcare, Technology, Financials, etc.)
- drhp_filing_date (YYYY-MM-DD or null)
- rhp_release_date (YYYY-MM-DD or null)
- ipo_open_date (YYYY-MM-DD or null)
- ipo_close_date (YYYY-MM-DD or null)
- shares_offered (e.g., 1,00,00,000 equity shares)
- ipo_total_size (e.g., ₹ 400 crore or ₹ 500 Cr)
- remarks (fresh issue vs offer for sale, book building etc.)
- rhp_release_month (e.g., Aug 2024)

Title: {title}
RHP Date (approx): {date_hint}

Only return valid JSON.
'''

def extract_with_openai(pdf_url, title, date_hint, client, model="gpt-4o"):
    try:
        # Step 1: Download PDF content
        pdf_response = requests.get(pdf_url, timeout=30)
        pdf_bytes = pdf_response.content

        # Step 2: Upload to OpenAI
        upload = client.files.create(
            file=pdf_bytes,
            purpose="assistants"
        )

        # Step 3: Send for extraction
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_prompt(title, date_hint)},
                        {"type": "file", "file_id": upload.id}
                    ],
                }
            ],
            response_format="json"
        )
        return json.loads(response.choices[0].message.content)

    except Exception as e:
        return {"error": f"Error: {str(e)}"}

# Execution
if start_btn:
    if not openai_key:
        st.error("Please enter your OpenAI API key.")
        st.stop()

    client = OpenAI(api_key=openai_key)
    st.info("Fetching RHP list from SEBI...")
    rhps = get_rhp_list(pages)
    output = []
    progress = st.progress(0)
    errors = []

    for i, rhp in enumerate(rhps):
        progress.progress((i + 1) / len(rhps))
        pdf_url = get_pdf_link(rhp["url"])
        if not pdf_url:
            errors.append(f"No PDF for {rhp['title']}")
            continue
        extraction = extract_with_openai(pdf_url, rhp["title"], rhp["date"], client, model)
        if "error" in extraction:
            errors.append(f"{rhp['title']}: {extraction['error']}")
            continue
        output.append({
            "Company Name": extraction.get("company_name"),
            "Sector": extraction.get("sector"),
            "DRHP Filing Date": extraction.get("drhp_filing_date"),
            "RHP Release Date": extraction.get("rhp_release_date"),
            "IPO Open Date": extraction.get("ipo_open_date"),
            "IPO Close Date": extraction.get("ipo_close_date"),
            "Shares offered": extraction.get("shares_offered"),
            "IPO Total Size": extraction.get("ipo_total_size"),
            "Remarks": extraction.get("remarks"),
            "RHP Release Month": extraction.get("rhp_release_month"),
            "PDF Link": pdf_url
        })
        time.sleep(1.5)

    if output:
        df = pd.DataFrame(output)
        st.success("✅ Extraction completed")
        st.dataframe(df, use_container_width=True)
        st.download_button("⬇️ Download CSV", df.to_csv(index=False), "sebi_rhp_data.csv", "text/csv")
    if errors:
        with st.expander("⚠️ Errors / Skipped"):
            for e in errors:
                st.error(e)
else:
    st.info("Enter API key and click Run to begin.")
