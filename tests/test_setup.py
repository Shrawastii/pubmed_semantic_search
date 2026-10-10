from src.config import API_KEY, EMAIL, QUERY
import requests

def test_env():
    assert API_KEY and EMAIL, "Fill in your .env file"

def test_ncbi_reachable():
    r = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"db": "pubmed", "term": QUERY, "retmax": 3,
                "retmode": "json", "api_key": API_KEY, "email": EMAIL},
        timeout=30)
    assert r.status_code == 200
    result = r.json()["esearchresult"]
    print("Total papers matching query:", result["count"])
    assert len(result["idlist"]) == 3

def test_model_loads():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-MiniLM-L6-v2")
    assert m.encode(["hello"]).shape[1] == 384