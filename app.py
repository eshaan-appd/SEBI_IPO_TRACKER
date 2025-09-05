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

RHP_TYPES_INCLUDE = ("- RHP", "RHP -", " – RHP", "— RHP", "RHP)")  # broad match
RHP_TYPES_EXCLUDE = ("Corrigendum", "Addendum", "Second Addendum")

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# ============================
# Helpers
# ============================

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    # Warm the site to set cookies/session
    try:
        s.get(SEBI_HOME, timeout=15)
    except Exception:
        pass
    return s


def _abs_url(base: str, href: str) -> str:
    return href if href.lower().startswith(("http://", "https://")) else urljoin(base, href)


def _parse_date(d: str) -> Optional[date]:
    if not d:
        return None
    try:
        # Examples on SEBI: "Aug 25, 2025"
        dt = dateparser.parse(d, dayfirst=False, yearfirst=False)
        return dt.date()
    except Exception:
        return None


def _month_name(d: Optional[date]) -> Optional[str]:
    return d.strftime("%b %Y") if d else None


def _looks_like_rhp(title: str) -> bool:
    t = title.strip()
    return any(x in t for x in RHP_TYPES_INCLUDE) and not any(x in t for x in RHP_TYPES_EXCLUDE)


def _company_from_title(title: str) -> str:
    # "Ather Energy Limited - RHP" -> "Ather Energy Limited"
    t = title.strip()
    t = re.sub(r"\s*[-–—]\s*RHP.*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*\(RHP\).*$", "", t, flags=re.IGNORECASE)
    return t.strip()


def _extract_pdf_links(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.select("a[href]"):
        href = a.get("href", "").strip()
        if not href:
            continue
        if href.lower().endswith(".pdf"):
            links.append(_abs_url(base_url, href))
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for u in links:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


def _list_page_url_rhp(page_no: int) -> str:
    # SEBI uses pno= for pagination on listing pages.
    # Keep other params stable.
    return f"{SEBI_PUBLIC_ISSUES_RHP}&pno={page_no}" if page_no > 1 else SEBI_PUBLIC_ISSUES_RHP


def _list_page_url_drhp(page_no: int) -> str:
    return f"{SEBI_PUBLIC_ISSUES_DRHP}&pno={page_no}" if page_no > 1 else SEBI_PUBLIC_ISSUES_DRHP


def _find_items_from_listing(html: str, base_url: str) -> List[Dict]:
    """Parse the standard SEBI listing table showing rows: Date | Title"""
    soup = BeautifulSoup(html, "lxml")
    items = []
    # Rows often appear inside <table> with <tr><td>Date</td><td><a href=...>Title</a></td></tr>
    # Fall back to scanning by link blocks near dates.
    rows = soup.select("table tr") or soup.select("div table tr")
    if not rows:
        # fallback: try list items
        rows = soup.select("li, div.row")
    for r in rows:
        tds = r.find_all(["td", "div"])
        if len(tds) < 2:
            continue
        # Heuristic: first cell contains a date like "Aug 25, 2025"
        date_txt = re.sub(r"\s+", " ", tds[0].get_text(strip=True))
        title_link = tds[1].find("a", href=True)
        if not title_link:
            continue
        title_txt = title_link.get_text(" ", strip=True)
        href = title_link["href"]
        if not title_txt or not date_txt:
            continue
        items.append(
            {
                "date_str": date_txt,
                "date": _parse_date(date_txt),
                "title": title_txt,
                "url": _abs_url(base_url, href),
            }
        )
    # Additional fallback: some pages are rendered as <a> lists with a date in a sibling node.
    if not items:
        for a in soup.select("a[href]"):
            title_txt = a.get_text(" ", strip=True)
            if not title_txt:
                continue
            # Try to find a date nearby
            parent_text = a.find_parent().get_text(" ", strip=True) if a.find_parent() else ""
            m = re.search(r"[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4}", parent_text)
            date_txt = m.group(0) if m else ""
            items.append(
                {
                    "date_str": date_txt,
                    "date": _parse_date(date_txt) if date_txt else None,
                    "title": title_txt,
                    "url": _abs_url(base_url, a["href"]),
                }
            )
    # Filter out obvious junk and duplicates
    uniq = []
    seen = set()
    for it in items:
        key = (it["date_str"], it["title"], it["url"])
        if key not in seen:
            uniq.append(it)
            seen.add(key)
    return uniq


# ============================
# OpenAI extraction
# ============================

def _get_openai_client(api_key: Optional[str] = None):
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("openai package is required. Add it to requirements.txt") from e
    key = api_key or os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY (set env var or add to Streamlit secrets, or enter it in the sidebar).")
    return OpenAI(api_key=key)


def _json_schema_spec() -> dict:
    # Constrain output to a deterministic JSON schema
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "RHPFields",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "company_name": {"type": "string"},
                    "sector": {"type": "string"},
                    "drhp_filing_date": {"type": ["string", "null"], "description": "ISO date YYYY-MM-DD or null"},
                    "rhp_release_date": {"type": ["string", "null"]},
                    "ipo_open_date": {"type": ["string", "null"]},
                    "ipo_close_date": {"type": ["string", "null"]},
                    "shares_offered": {"type": ["string", "null"], "description": "Total equity shares offered; include units if ambiguous"},
                    "ipo_total_size": {"type": ["string", "null"], "description": "Aggregate size; prefer INR ₹ crore with unit"},
                    "remarks": {"type": ["string", "null"], "description": "Fresh/OFS mix, notable notes"},
                    "rhp_release_month": {"type": ["string", "null"], "description": "Mon YYYY e.g., Aug 2025"}
                },
                "required": ["company_name", "sector", "remarks"],
            },
            "strict": True,
        },
    }


def _extraction_prompt(company_hint: str, rhp_date_hint_iso: Optional[str]) -> str:
    return (
        "You are an expert analyst reading an Indian IPO Red Herring Prospectus (RHP). "
        "Extract the following fields from THIS document only. If a field is not explicitly present or contains placeholders like [●], return null.\n\n"
        "Fields:\n"
        "- company_name\n"
        "- sector (concise single label like one of: Automobiles, Capital Goods, Consumer Discretionary, Consumer Staples, Energy, Financials, Healthcare, Industrials, Materials, Real Estate, Technology, Telecom, Utilities, Other)\n"
        "- drhp_filing_date (YYYY-MM-DD if explicitly mentioned; else null)\n"
        "- rhp_release_date (if printed or infer from cover; else null)\n"
        "- ipo_open_date (if present; else null)\n"
        "- ipo_close_date (if present; else null)\n"
        "- shares_offered (total equity shares offered across fresh + OFS, with units if ambiguous)\n"
        "- ipo_total_size (₹ crore if available; else null)\n"
        "- remarks (fresh issue vs offer for sale mix, book building/fixed price, any notable features)\n"
        "- rhp_release_month (Mon YYYY)\n\n"
        f"Use this hint for company if needed: {company_hint or 'N/A'}. "
        f"Use this hint for RHP date if needed: {rhp_date_hint_iso or 'N/A'}.\n"
        "Return ONLY valid JSON that matches the provided JSON schema."
    )


class TemporaryRateLimit(Exception):
    pass


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type(TemporaryRateLimit),
)
def _extract_with_openai(client, model: str, pdf_url: str, company_hint: str, rhp_date_hint: Optional[date]) -> dict:
    rhp_date_iso = rhp_date_hint.isoformat() if rhp_date_hint else None

    prompt = _extraction_prompt(company_hint, rhp_date_iso)
    schema = _json_schema_spec()

    try:
        resp = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        # Use direct PDF URL so OpenAI reads the PDF itself (no local parsing).
                        {"type": "input_file", "file_url": pdf_url},
                    ],
                }
            ],
            response_format=schema,
        )
    except Exception as e:
        # Simple heuristic to re-try on 429s
        if "rate" in str(e).lower() or "429" in str(e):
            raise TemporaryRateLimit(e)
        raise

    # The SDK exposes output_text which is guaranteed to match the schema requested.
    try:
        output = resp.output_text  # type: ignore[attr-defined]
        return json.loads(output)
    except Exception:
        # Fallback: try to find first JSON object in the response content
        try:
            text = getattr(resp, "output_text", None) or ""
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            return json.loads(m.group(0)) if m else {}
        except Exception:
            return {}


# ============================
# DRHP date backfill via listing (optional)
# ============================

def _search_drhp_listing_for_company(s: requests.Session, company: str, max_pages: int = 5) -> Optional[date]:
    """Heuristic: scan first N pages of Draft Offer Documents listing to find the company's DRHP date."""
    cname_norm = re.sub(r"\s+", " ", company.lower()).strip()
    for p in range(1, max_pages + 1):
        url = _list_page_url_drhp(p)
        try:
            r = s.get(url, timeout=30)
            r.raise_for_status()
        except Exception:
            continue
        items = _find_items_from_listing(r.text, url)
        for it in items:
            title_norm = re.sub(r"\s+", " ", it.get("title", "").lower()).strip()
            if cname_norm and cname_norm.split()[0] in title_norm and "drhp" in title_norm:
                return it.get("date")
    return None


# ============================
# Streamlit UI
# ============================

st.set_page_config(page_title="SEBI RHP Extractor (OpenAI-powered)", layout="wide")

st.title("📄 SEBI RHP Extractor — FILINGS ▸ Public Issues ▸ RHP (Filed with RoC)")

with st.expander("What this app does"):
    st.markdown(
        """
- Crawls SEBI → **Filings → Public Issues → Red Herring Documents filed with RoC** (RHP list).  
- Opens each RHP detail page, fetches the attached **PDF** link(s).  
- Sends the PDF **directly** to OpenAI using *file URL inputs* and extracts structured fields.  
- Outputs a table with: **Company, Sector, DRHP Filing Date, RHP Release Date, IPO Open/Close, Shares Offered, IPO Total Size, Remarks, RHP Release Month**.  
        """
    )

with st.sidebar:
    st.header("Settings")
    api_key_input = st.text_input("🔑 OpenAI API Key (sk-…)", type="password", help="If empty, the app will use OPENAI_API_KEY env or secrets.")
    model = st.text_input("🧠 OpenAI model", value=DEFAULT_MODEL, help="e.g., gpt-4.1-mini / o4-mini (supports file inputs)")
    max_pages = st.number_input("Pages to scan (RHP listing)", min_value=1, max_value=50, value=3, step=1)
    include_addenda = st.checkbox("Include Addendum/Corrigendum rows", value=False)
    drhp_backfill = st.checkbox("Backfill DRHP date via DRHP listing (heuristic)", value=True)
    date_from = st.date_input("From date", value=date(date.today().year, 1, 1))
    date_to = st.date_input("To date", value=date.today())

    run_btn = st.button("🚀 Run Extraction")

progress = st.empty()
log = st.empty()
out_df_placeholder = st.empty()

@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_listing_page(session: requests.Session, page_no: int) -> Tuple[str, List[Dict]]:
    url = _list_page_url_rhp(page_no)
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    items = _find_items_from_listing(resp.text, url)
    return url, items

@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_detail_page(session: requests.Session, url: str) -> Tuple[str, List[str]]:
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    pdfs = _extract_pdf_links(resp.text, url)
    return resp.text, pdfs

def _filter_items(items: List[Dict], df: date, dt: date, include_addenda: bool) -> List[Dict]:
    kept = []
    for it in items:
        title = it.get("title", "")
        if not include_addenda and any(x.lower() in title.lower() for x in ["corrigendum", "addendum"]):
            continue
        if not _looks_like_rhp(title):
            continue
        d = it.get("date")
        if d and (d < df or d > dt):
            continue
        kept.append(it)
    return kept

def _make_row_from_extraction(ex: dict, fallback_company: str, rhp_date: Optional[date]) -> Dict[str, Optional[str]]:
    company = ex.get("company_name") or fallback_company
    rhp_iso = ex.get("rhp_release_date") or (rhp_date.isoformat() if rhp_date else None)
    row = {
        "Company Name": company,
        "Sector": ex.get("sector"),
        "DRHP Filing Date": ex.get("drhp_filing_date"),
        "RHP Release Date": rhp_iso,
        "IPO Open Date": ex.get("ipo_open_date"),
        "IPO Close Date": ex.get("ipo_close_date"),
        "Shares offered": ex.get("shares_offered"),
        "IPO Total Size": ex.get("ipo_total_size"),
        "Remarks": ex.get("remarks"),
        "RHP Release Month": ex.get("rhp_release_month") or (_month_name(rhp_date) if rhp_date else None),
    }
    return row

if run_btn:
    s = _session()
    rows: List[Dict] = []
    errors: List[str] = []
    total_scanned = 0

    client = None
    try:
        client = _get_openai_client(api_key_input)
    except Exception as e:
        st.error(str(e))
        st.stop()

    for page in range(1, int(max_pages) + 1):
        try:
            list_url, items = _fetch_listing_page(s, page)
        except Exception as e:
            errors.append(f"Failed listing page {page}: {e}")
            continue

        items = _filter_items(items, date_from, date_to, include_addenda)
        progress.info(f"Page {page}: {len(items)} RHP rows after filtering")

        for it in items:
            total_scanned += 1
            title = it.get("title", "")
            rhp_date = it.get("date")
            detail_url = it.get("url", "")
            company_hint = _company_from_title(title)

            log.write(f"🔎 [{total_scanned}] {company_hint} | {rhp_date} — {detail_url}")

            # Fetch detail page and collect PDFs
            try:
                html, pdfs = _fetch_detail_page(s, detail_url)
            except Exception as e:
                errors.append(f"[{company_hint}] detail fetch error: {e}")
                continue

            if not pdfs:
                errors.append(f"[{company_hint}] No PDFs found on detail page")
                continue

            # Prefer first non-addendum PDF if multiple
            main_pdf = None
            for purl in pdfs:
                if ("addendum" in purl.lower()) or ("corrigendum" in purl.lower()):
                    continue
                main_pdf = purl
                break
            if not main_pdf:
                main_pdf = pdfs[0]

            # Extract with OpenAI (direct file URL)
            try:
                ex = _extract_with_openai(client, model, main_pdf, company_hint, rhp_date)
            except Exception as e:
                errors.append(f"[{company_hint}] OpenAI extraction error: {e}")
                continue

            # Optional DRHP backfill if missing
            if drhp_backfill and (not ex.get("drhp_filing_date") or ex.get("drhp_filing_date") in ("", None)):
                try:
                    drhp_dt = _search_drhp_listing_for_company(s, company_hint, max_pages=3)
                    if drhp_dt:
                        ex["drhp_filing_date"] = drhp_dt.isoformat()
                except Exception:
                    pass

            row = _make_row_from_extraction(ex, company_hint, rhp_date)
            rows.append(row)

            # Gentle delay to be kind to both SEBI and OpenAI
            time.sleep(1.5)

    if rows:
        df = pd.DataFrame(rows).drop_duplicates()
        out_df_placeholder.dataframe(df, use_container_width=True)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="sebi_rhp_extracted.csv", mime="text/csv")
        try:
            xlsx_bytes = df.to_excel(index=False, engine="xlsxwriter")
        except Exception:
            # Fallback without engine if unavailable
            xlsx_bytes = df.to_excel(index=False)
        # Streamlit requires bytes; to_excel returns bytes if engine writes to buffer, else raise.
        st.success(f"Done. Extracted {len(df)} rows from {total_scanned} scanned.")
    else:
        st.warning("No rows extracted for the chosen filters. Try increasing pages, widening dates, or toggling addendum inclusion.")

    if errors:
        with st.expander("⚠️ Warnings / Errors"):
            for e in errors:
                st.write("- " + str(e))
else:
    st.info("Set your API key and click **Run Extraction** to begin.")
