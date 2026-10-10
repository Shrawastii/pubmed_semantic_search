import time, sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from src.config import API_KEY, EMAIL, QUERY

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
BATCH = 200
SLEEP = 0.11 if API_KEY else 0.34
MIN_WORDS = 30

DATA_DIR = Path("data")
OUT_FILE = DATA_DIR / "abstracts.parquet"


def get_pmids(query):
    r = requests.get(BASE + "esearch.fcgi", params={
        "db": "pubmed", "term": query, "retmax": 10000,
        "retmode": "json", "api_key": API_KEY, "email": EMAIL}, timeout=60)
    r.raise_for_status()
    res = r.json()["esearchresult"]
    count, ids = int(res["count"]), res["idlist"]
    print(f"PubMed reports {count} matches, retrieved {len(ids)} PMIDs")
    if count > len(ids):
        print("WARNING: more than 10,000 matches. Narrow the date range.")
    return ids


def fetch_batch(pmids, retries=3):
    for attempt in range(retries):
        try:
            r = requests.post(BASE + "efetch.fcgi", data={
                "db": "pubmed", "id": ",".join(pmids), "retmode": "xml",
                "api_key": API_KEY, "email": EMAIL}, timeout=120)
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            print(f"Batch failed ({e}), retry {attempt + 1}/{retries}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("Batch failed after retries")


def text_of(elem):
    return "".join(elem.itertext()).strip() if elem is not None else ""


def parse(xml_text):
    rows = []
    root = ET.fromstring(xml_text)
    for art in root.findall(".//PubmedArticle"):
        pmid = text_of(art.find("./MedlineCitation/PMID"))
        title = text_of(art.find(".//Article/ArticleTitle"))
        # abstracts can be split into labeled sections, so join them
        abstract = " ".join(
            text_of(a) for a in art.findall(".//Article/Abstract/AbstractText")
        ).strip()
        journal = text_of(art.find(".//Article/Journal/Title"))
        pub = art.find(".//Article/Journal/JournalIssue/PubDate")
        year = ""
        if pub is not None:
            year = text_of(pub.find("Year")) or text_of(pub.find("MedlineDate"))[:4]
        mesh = "; ".join(
            text_of(m) for m in art.findall(".//MeshHeadingList/MeshHeading/DescriptorName")
        )
        rows.append({"pmid": pmid, "title": title, "abstract": abstract,
                     "year": year, "journal": journal, "mesh": mesh})
    return rows


def main(force=False):
    DATA_DIR.mkdir(exist_ok=True)
    if OUT_FILE.exists() and not force:
        print(f"{OUT_FILE} already exists. Use --force to download again.")
        return

    pmids = get_pmids(QUERY)
    (DATA_DIR / "pmids.txt").write_text("\n".join(pmids))

    rows = []
    for i in tqdm(range(0, len(pmids), BATCH), desc="Downloading"):
        rows += parse(fetch_batch(pmids[i:i + BATCH]))
        time.sleep(SLEEP)

    df = pd.DataFrame(rows).drop_duplicates("pmid")
    n_all = len(df)
    df["n_words"] = df["abstract"].str.split().str.len()
    df = df[df["n_words"] >= MIN_WORDS].reset_index(drop=True)
    df["text"] = df["title"] + ". " + df["abstract"]

    df.to_parquet(OUT_FILE, index=False)
    print(f"Fetched {n_all} records, kept {len(df)} with abstracts of {MIN_WORDS}+ words")
    print(f"Saved to {OUT_FILE}")
    print(df[["pmid", "year", "n_words"]].describe(include="all").loc[["count", "min", "max"]])


if __name__ == "__main__":
    main(force="--force" in sys.argv)
    