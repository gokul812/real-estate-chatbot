"""
Mumbai Real Estate AI — FastAPI backend
Streams responses via SSE with full tool-call loop.
Exclusively covers Mumbai: Western Line, Central Line, Harbour Line & South Mumbai.
"""

import json
import os
import httpx as _httpx
from groq import AsyncGroq

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Any

client = AsyncGroq()

# ─── Mumbai Zone Definitions ──────────────────────────────────────────────────

# All Western Railway stations Churchgate → Virar (south to north)
WESTERN_LINE = [
    # South Mumbai terminus → mid suburbs
    "churchgate", "marine lines", "charni road", "grant road", "mumbai central",
    "mahalaxmi", "lower parel", "elphinstone", "prabhadevi", "dadar west",
    "matunga road", "mahim", "bandra", "khar road", "khar", "santacruz",
    "vile parle", "andheri", "jogeshwari", "goregaon", "malad", "kandivali",
    "borivali", "dahisar",
    # Mira-Bhayander & beyond (Mumbai Metropolitan Region, Western Railway)
    "mira road", "mira-bhayandar", "bhayander", "bhayandar",
    "naigaon", "vasai road", "vasai", "nallasopara", "nalasopara", "virar",
    # Notable sub-areas along Western line
    "juhu", "versova", "lokhandwala", "oshiwara", "four bungalows",
    "vasai-virar", "virar west", "virar east",
]

CENTRAL_LINE = [
    "csmt", "byculla", "chinchpokli", "currey road", "parel", "dadar",
    "matunga", "sion", "kurla", "vidyavihar", "ghatkopar", "vikhroli",
    "kanjurmarg", "bhandup", "mulund", "chembur", "tilak nagar",
    "govandi", "mankhurd", "chunabhatti", "powai", "hiranandani",
]

HARBOUR_LINE = [
    "wadala", "king circle", "mahim junction", "sewri", "cotton green",
    "reay road", "dockyard road", "mazgaon", "sandhurst road", "mandala",
]

SOUTH_MUMBAI = [
    "colaba", "cuffe parade", "nariman point", "fort", "ballard estate",
    "marine drive", "malabar hill", "worli", "breach candy", "pedder road",
    "walkeshwar", "bhuleshwar", "nepean sea road", "altamount road",
]

MUMBAI_ZONES = list(set(
    WESTERN_LINE + CENTRAL_LINE + HARBOUR_LINE + SOUTH_MUMBAI
    + ["mumbai", "bombay", "bkc", "bandra kurla complex", "lower parel", "worli"]
))

WALKSCORE_API_KEY  = os.environ.get("WALKSCORE_API_KEY", "")
RAPIDAPI_KEY       = os.environ.get("RAPIDAPI_KEY", "")
# Set RAPIDAPI_INDIA_HOST to any free India property API host on rapidapi.com
# e.g. "indian-property-listing.p.rapidapi.com" or "housing-com.p.rapidapi.com"
RAPIDAPI_INDIA_HOST = os.environ.get("RAPIDAPI_INDIA_HOST", "indian-property-listing.p.rapidapi.com")


def is_mumbai_location(location: str) -> bool:
    loc = location.lower()
    return any(k in loc for k in MUMBAI_ZONES)


def _zone_for(loc: str) -> str:
    loc = loc.lower()
    if any(k in loc for k in WESTERN_LINE):
        return "western"
    if any(k in loc for k in CENTRAL_LINE):
        return "central"
    if any(k in loc for k in HARBOUR_LINE):
        return "harbour"
    if any(k in loc for k in SOUTH_MUMBAI):
        return "south"
    return "western"


# ─── Tool Definitions ─────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "search_properties",
        "description": (
            "Search for properties in Mumbai. Covers the full Western Line (Churchgate to Virar — "
            "Bandra, Andheri, Juhu, Malad, Borivali, Mira Road, Vasai, Nallasopara, Virar), "
            "Central Line (Dadar, Kurla, Ghatkopar, Powai, Mulund, Chembur), "
            "Harbour Line (Wadala, Sewri, Chembur), and South Mumbai "
            "(Colaba, Worli, Lower Parel, Prabhadevi, Malabar Hill). "
            "Returns live listings when available, else realistic demo data. All prices in INR."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Mumbai neighborhood or area (e.g. 'Bandra West', 'Powai', 'Worli', 'Chembur')"
                },
                "min_price": {
                    "type": "number",
                    "description": "Minimum price in INR (e.g. 5000000 for 50 lakhs, 10000000 for 1 crore)"
                },
                "max_price": {
                    "type": "number",
                    "description": "Maximum price in INR (e.g. 30000000 for 3 crore)"
                },
                "bedrooms": {
                    "type": "integer",
                    "description": "Minimum number of bedrooms/BHK"
                },
                "bathrooms": {
                    "type": "number",
                    "description": "Minimum number of bathrooms"
                },
                "property_type": {
                    "type": "string",
                    "enum": ["flat", "apartment", "house", "villa", "penthouse", "studio", "plot", "any"],
                    "description": "Type of property"
                },
                "for_sale_or_rent": {
                    "type": "string",
                    "enum": ["sale", "rent", "both"],
                    "description": "Whether looking to buy or rent"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "calculate_emi",
        "description": (
            "Calculate monthly EMI, total interest, and complete home loan details for Mumbai properties. "
            "All calculations in INR. Includes Indian bank rates, Section 24b/80C tax benefits, "
            "stamp duty guidance, and processing fee estimates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "home_price": {
                    "type": "number",
                    "description": "Property price in INR (e.g. 15000000 for 1.5 crore)"
                },
                "down_payment_percent": {
                    "type": "number",
                    "description": "Down payment as percentage (minimum 10-20% in India)"
                },
                "annual_interest_rate": {
                    "type": "number",
                    "description": "Annual interest rate %. SBI ~8.50%, HDFC ~8.70%, ICICI ~8.75%"
                },
                "loan_term_years": {
                    "type": "integer",
                    "description": "Loan tenure in years (up to 30 years)"
                },
                "property_tax_annual": {
                    "type": "number",
                    "description": "Annual property tax in INR (optional)"
                }
            },
            "required": ["home_price", "down_payment_percent", "annual_interest_rate", "loan_term_years"]
        }
    },
    {
        "name": "get_neighborhood_info",
        "description": (
            "Get detailed information about a Mumbai neighborhood: schools, hospitals, "
            "metro/local train connectivity, walkability, real estate prices, and lifestyle. "
            "Covers all Western, Central, Harbour, and South Mumbai areas."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "neighborhood": {
                    "type": "string",
                    "description": "Mumbai neighborhood name (e.g. 'Bandra', 'Powai', 'Chembur', 'Worli')"
                }
            },
            "required": ["neighborhood"]
        }
    },
    {
        "name": "get_market_trends",
        "description": (
            "Get Mumbai real estate market trends: price changes, new launches, absorption rates, "
            "top performing micro-markets, and forecast. Can be filtered by zone "
            "(western/central/harbour/south) or specific area."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Mumbai area or zone (e.g. 'Bandra', 'Western Mumbai', 'Mumbai')"
                },
                "period": {
                    "type": "string",
                    "enum": ["3months", "6months", "1year", "3years"],
                    "description": "Time period for trend data"
                }
            },
            "required": ["location"]
        }
    }
]

# Convert to Groq/OpenAI format
GROQ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in TOOLS
]


# ─── Live API helpers ──────────────────────────────────────────────────────────

def _geocode(address: str):
    """Lat/lon via OpenStreetMap Nominatim — free, no key needed."""
    try:
        r = _httpx.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": address + ", Mumbai, India", "format": "json", "limit": 1},
            headers={"User-Agent": "MumbaiRealEstateChatbot/2.0"},
            timeout=6,
        )
        data = r.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass
    return None


def _walk_score(address: str, lat: float, lon: float):
    """Walk/Transit/Bike scores — free API key from walkscore.com."""
    if not WALKSCORE_API_KEY:
        return None
    try:
        r = _httpx.get(
            "https://api.walkscore.com/score",
            params={
                "format": "json",
                "address": address,
                "lat": lat,
                "lon": lon,
                "transit": 1,
                "bike": 1,
                "wsapikey": WALKSCORE_API_KEY,
            },
            timeout=6,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ─── Live property listing fetcher ───────────────────────────────────────────
# Uses RapidAPI free tier (50–500 req/month depending on plan).
# Sign up free at https://rapidapi.com → search "India Real Estate" or "Mumbai Properties"
# Set env vars: RAPIDAPI_KEY and optionally RAPIDAPI_INDIA_HOST

def _fetch_live_properties(location: str, bedrooms=None, min_price=None,
                           max_price=None, for_rent=False):
    """
    Tries two common RapidAPI India property endpoints.
    Returns a normalised list of listings, or None if unavailable.
    """
    if not RAPIDAPI_KEY:
        return None

    # Endpoint candidates — tries each in order until one succeeds
    # Each tuple: (host, path, param_map)
    candidates = [
        {
            "host": RAPIDAPI_INDIA_HOST,
            "url":  f"https://{RAPIDAPI_INDIA_HOST}/properties",
            "params": {
                "city":     "Mumbai",
                "location": location,
                "purpose":  "for-rent" if for_rent else "for-sale",
                "rooms":    str(bedrooms) if bedrooms else None,
                "priceMin": str(int(min_price)) if min_price else None,
                "priceMax": str(int(max_price)) if max_price else None,
                "hitsPerPage": "10",
            },
        },
        {
            "host": "indian-real-estate4.p.rapidapi.com",
            "url":  "https://indian-real-estate4.p.rapidapi.com/search",
            "params": {
                "location": f"{location} Mumbai",
                "type":     "rent" if for_rent else "buy",
                "limit":    "10",
            },
        },
    ]

    for cand in candidates:
        try:
            params = {k: v for k, v in cand["params"].items() if v is not None}
            resp = _httpx.get(
                cand["url"],
                headers={
                    "X-RapidAPI-Key":  RAPIDAPI_KEY,
                    "X-RapidAPI-Host": cand["host"],
                },
                params=params,
                timeout=8,
            )
            if resp.status_code != 200:
                continue

            raw = resp.json()
            # Normalise: API may return list or dict with hits/results key
            hits = raw
            if isinstance(raw, dict):
                for key in ("hits", "results", "data", "properties", "listings"):
                    if key in raw:
                        hits = raw[key]
                        break

            if not isinstance(hits, list) or not hits:
                continue

            listings = []
            for i, p in enumerate(hits[:8]):
                price = (p.get("price") or p.get("priceValue") or
                         p.get("listingPrice") or p.get("amount") or 0)
                try:
                    price = int(float(str(price).replace(",", "").replace("₹", "")))
                except (ValueError, TypeError):
                    price = 0

                beds  = p.get("bedrooms") or p.get("bhk") or p.get("rooms") or "N/A"
                baths = p.get("bathrooms") or p.get("bath") or "N/A"
                sqft  = p.get("area") or p.get("carpetArea") or p.get("superArea") or p.get("squareFootage") or "N/A"
                ptype = (p.get("type") or p.get("propertyType") or "flat").lower()
                addr  = (p.get("address") or p.get("title") or p.get("name")
                         or p.get("locality") or f"Property in {location}, Mumbai")

                listings.append({
                    "id":            p.get("id", f"LIVE-{i+1}"),
                    "address":       addr,
                    "price":         price,
                    "price_display": _inr_fmt(price) + ("/month" if for_rent else ""),
                    "bedrooms":      beds,
                    "bathrooms":     baths,
                    "sqft":          sqft,
                    "type":          ptype,
                    "status":        "for rent" if for_rent else "for sale",
                    "days_on_market": p.get("daysOnMarket") or p.get("postedDaysAgo", "N/A"),
                    "floor":         p.get("floor") or p.get("floorNumber", "N/A"),
                    "society":       p.get("society") or p.get("project") or p.get("building", ""),
                    "rera_id":       p.get("reraId") or p.get("rera"),
                    "features":      p.get("amenities") or p.get("features") or [],
                    "description":   p.get("description") or "",
                    "source":        f"Live — {cand['host']}",
                })

            if listings:
                return listings

        except Exception:
            continue

    return None


# ─── Area price multipliers ────────────────────────────────────────────────────

AREA_CONFIG = {
    # South Mumbai — ultra premium
    "colaba":         {"label": "Colaba",               "mult": 3.5,  "zone": "south"},
    "cuffe parade":   {"label": "Cuffe Parade",          "mult": 3.8,  "zone": "south"},
    "nariman point":  {"label": "Nariman Point",         "mult": 3.2,  "zone": "south"},
    "malabar hill":   {"label": "Malabar Hill",          "mult": 4.0,  "zone": "south"},
    "altamount road": {"label": "Altamount Road",        "mult": 4.5,  "zone": "south"},
    "pedder road":    {"label": "Pedder Road",           "mult": 3.6,  "zone": "south"},
    "nepean sea road":{"label": "Nepean Sea Road",       "mult": 3.4,  "zone": "south"},
    "breach candy":   {"label": "Breach Candy",          "mult": 3.0,  "zone": "south"},
    # Prabhadevi / Lower Parel / Worli corridor
    "worli":          {"label": "Worli",                 "mult": 2.8,  "zone": "south"},
    "prabhadevi":     {"label": "Prabhadevi",            "mult": 2.2,  "zone": "south"},
    "lower parel":    {"label": "Lower Parel",           "mult": 2.4,  "zone": "south"},
    "mahalaxmi":      {"label": "Mahalaxmi",             "mult": 2.0,  "zone": "western"},
    # BKC adjacent
    "bkc":            {"label": "Bandra Kurla Complex",  "mult": 2.6,  "zone": "western"},
    # Western line
    "bandra":         {"label": "Bandra West",           "mult": 2.2,  "zone": "western"},
    "khar":           {"label": "Khar West",             "mult": 1.9,  "zone": "western"},
    "santacruz":      {"label": "Santacruz West",        "mult": 1.7,  "zone": "western"},
    "vile parle":     {"label": "Vile Parle West",       "mult": 1.6,  "zone": "western"},
    "juhu":           {"label": "Juhu",                  "mult": 2.5,  "zone": "western"},
    "versova":        {"label": "Versova",               "mult": 1.5,  "zone": "western"},
    "andheri":        {"label": "Andheri West",          "mult": 1.4,  "zone": "western"},
    "lokhandwala":    {"label": "Lokhandwala, Andheri",  "mult": 1.5,  "zone": "western"},
    "jogeshwari":     {"label": "Jogeshwari West",       "mult": 1.2,  "zone": "western"},
    "goregaon":       {"label": "Goregaon West",         "mult": 1.1,  "zone": "western"},
    "malad":          {"label": "Malad West",            "mult": 1.0,  "zone": "western"},
    "kandivali":      {"label": "Kandivali West",        "mult": 0.9,  "zone": "western"},
    "borivali":       {"label": "Borivali West",         "mult": 0.85, "zone": "western"},
    "dahisar":        {"label": "Dahisar",               "mult": 0.70, "zone": "western"},
    # Western line — Mira-Bhayander corridor
    "mira road":      {"label": "Mira Road",             "mult": 0.58, "zone": "western"},
    "mira-bhayandar": {"label": "Mira-Bhayandar",        "mult": 0.53, "zone": "western"},
    "bhayander":      {"label": "Bhayandar West",        "mult": 0.50, "zone": "western"},
    "bhayandar":      {"label": "Bhayandar West",        "mult": 0.50, "zone": "western"},
    # Western line — Vasai-Virar belt
    "naigaon":        {"label": "Naigaon",               "mult": 0.38, "zone": "western"},
    "vasai road":     {"label": "Vasai Road",            "mult": 0.42, "zone": "western"},
    "vasai":          {"label": "Vasai West",            "mult": 0.40, "zone": "western"},
    "nallasopara":    {"label": "Nallasopara West",      "mult": 0.32, "zone": "western"},
    "nalasopara":     {"label": "Nalasopara West",       "mult": 0.32, "zone": "western"},
    "virar":          {"label": "Virar West",            "mult": 0.36, "zone": "western"},
    "virar west":     {"label": "Virar West",            "mult": 0.36, "zone": "western"},
    "virar east":     {"label": "Virar East",            "mult": 0.30, "zone": "western"},
    # Western line mid-stations
    "khar road":      {"label": "Khar Road",             "mult": 1.85, "zone": "western"},
    "matunga road":   {"label": "Matunga Road",          "mult": 1.55, "zone": "western"},
    # Central line
    "dadar":          {"label": "Dadar West",            "mult": 1.8,  "zone": "central"},
    "matunga":        {"label": "Matunga",               "mult": 1.6,  "zone": "central"},
    "sion":           {"label": "Sion",                  "mult": 1.3,  "zone": "central"},
    "kurla":          {"label": "Kurla",                 "mult": 1.1,  "zone": "central"},
    "ghatkopar":      {"label": "Ghatkopar West",        "mult": 1.2,  "zone": "central"},
    "vikhroli":       {"label": "Vikhroli",              "mult": 1.0,  "zone": "central"},
    "powai":          {"label": "Powai",                 "mult": 1.4,  "zone": "central"},
    "hiranandani":    {"label": "Hiranandani, Powai",    "mult": 1.5,  "zone": "central"},
    "kanjurmarg":     {"label": "Kanjurmarg",            "mult": 1.0,  "zone": "central"},
    "bhandup":        {"label": "Bhandup West",          "mult": 0.9,  "zone": "central"},
    "mulund":         {"label": "Mulund West",           "mult": 0.95, "zone": "central"},
    "chembur":        {"label": "Chembur",               "mult": 1.3,  "zone": "central"},
    "tilak nagar":    {"label": "Tilak Nagar",           "mult": 1.1,  "zone": "central"},
    "govandi":        {"label": "Govandi",               "mult": 0.7,  "zone": "central"},
    "mankhurd":       {"label": "Mankhurd",              "mult": 0.65, "zone": "central"},
    # Harbour line
    "wadala":         {"label": "Wadala",                "mult": 1.3,  "zone": "harbour"},
    "sewri":          {"label": "Sewri",                 "mult": 0.9,  "zone": "harbour"},
    "mazgaon":        {"label": "Mazgaon",               "mult": 1.0,  "zone": "harbour"},
    "byculla":        {"label": "Byculla",               "mult": 1.1,  "zone": "central"},
    "parel":          {"label": "Parel",                 "mult": 1.9,  "zone": "central"},
}

BASE_PRICE = 10_000_000  # 1 Cr base


def _area_cfg(location: str) -> dict:
    loc = location.lower()
    for key, cfg in AREA_CONFIG.items():
        if key in loc:
            return cfg
    return {"label": "Mumbai", "mult": 1.0, "zone": "western"}


# ─── Area-Specific Property Database ─────────────────────────────────────────
# Every listing is tied to the EXACT area — real builders, real projects,
# accurate 2024 prices per sqft for that micro-market.

def _inr_fmt(v: int) -> str:
    if v >= 10_000_000:
        cr = v / 10_000_000
        return f"Rs. {cr:.2f} Cr"
    elif v >= 100_000:
        l = v / 100_000
        return f"Rs. {l:.1f} L"
    else:
        return f"Rs. {v:,}"

def _p(price): return {"price": price, "price_display": _inr_fmt(price)}
def _r(rent):  return {"price": rent,  "price_display": _inr_fmt(rent) + "/month", "status": "for rent"}

AREA_LISTINGS: dict = {

    # ── VIRAR ─────────────────────────────────────────────────────────────────
    "virar": [
        {"id":"VR-01","address":"Sunteck City Avenue 1, Wing B, Flat 502, Virar West, Mumbai-401303",
         "bedrooms":2,"bathrooms":2,"sqft":790,"carpet_area_sqft":595,"type":"flat","status":"for sale",
         "floor":"5th of 20","society":"Sunteck City","possession":"Ready to Move","rera_id":"P99000010832",
         "price_per_sqft":"Rs. 5,800/sqft","days_on_market":18,"zone":"western",
         "description":"2 BHK in Sunteck City Avenue 1, Virar West — gated township with clubhouse, near Virar railway station.",
         "features":["modular kitchen","covered parking","24hr security","power backup"],
         "amenities":["clubhouse","swimming pool","gymnasium","children's play area"],"source":"demo",
         **_p(4_582_000)},
        {"id":"VR-02","address":"Bhagwati Greens Phase 2, Tower C, Apt 801, Virar West, Mumbai-401303",
         "bedrooms":1,"bathrooms":1,"sqft":550,"carpet_area_sqft":415,"type":"flat","status":"for sale",
         "floor":"8th of 15","society":"Bhagwati Greens","possession":"Ready to Move","rera_id":"P99000008741",
         "price_per_sqft":"Rs. 5,200/sqft","days_on_market":9,"zone":"western",
         "description":"Compact 1 BHK in Bhagwati Greens, Virar West — ideal first home or investment near Virar station.",
         "features":["modular kitchen","power backup","CCTV","intercom"],
         "amenities":["gym","garden","24hr security"],"source":"demo",
         **_p(2_860_000)},
        {"id":"VR-03","address":"Sheth Vasant Vihar, Flat 304, Agashi Road, Virar West, Mumbai-401303",
         "bedrooms":2,"bathrooms":2,"sqft":900,"carpet_area_sqft":678,"type":"flat","status":"for sale",
         "floor":"3rd of 12","society":"Sheth Vasant Vihar","possession":"Under Construction — Mar 2026","rera_id":"P99000012154",
         "price_per_sqft":"Rs. 6,200/sqft","days_on_market":31,"zone":"western",
         "description":"2 BHK by Sheth Creators in Virar West with Vasai creek views. Gated complex 5 min from Virar station.",
         "features":["creek view","covered parking","power backup","vastu compliant"],
         "amenities":["rooftop garden","gym","swimming pool"],"source":"demo",
         **_p(5_580_000)},
        {"id":"VR-04","address":"Hari Om CHS, Flat 201, Station Road, Virar East, Mumbai-401305",
         "bedrooms":2,"bathrooms":1,"sqft":750,"carpet_area_sqft":562,"type":"flat","status":"for rent",
         "floor":"2nd of 7","society":"Hari Om CHS","possession":"Immediate","rera_id":None,
         "price_per_sqft":"Rs. 12/sqft/month","days_on_market":5,"zone":"western",
         "description":"Semi-furnished 2 BHK on rent in Virar East, walking distance to Virar railway station.",
         "features":["semi-furnished","fan/light fittings","2 bathrooms","storage loft"],
         "amenities":["watchman","water storage"],"source":"demo",
         **_r(9_000)},
    ],

    # ── NALLASOPARA ───────────────────────────────────────────────────────────
    "nallasopara": [
        {"id":"NS-01","address":"Shiv Shakti Residency, Wing A, Flat 603, Nallasopara West, Mumbai-401203",
         "bedrooms":2,"bathrooms":2,"sqft":860,"carpet_area_sqft":645,"type":"flat","status":"for sale",
         "floor":"6th of 14","society":"Shiv Shakti Residency","possession":"Ready to Move","rera_id":"P99000006521",
         "price_per_sqft":"Rs. 4,800/sqft","days_on_market":22,"zone":"western",
         "description":"2 BHK in Shiv Shakti Residency, Nalasopara West — gated complex near Nalasopara railway station.",
         "features":["modular kitchen","covered parking","CCTV","power backup"],
         "amenities":["gym","children play area","garden","24hr security"],"source":"demo",
         **_p(4_128_000)},
        {"id":"NS-02","address":"Gurukrupa Galaxy, Flat 404, Manor Road, Nallasopara East, Mumbai-401209",
         "bedrooms":1,"bathrooms":1,"sqft":480,"carpet_area_sqft":360,"type":"flat","status":"for sale",
         "floor":"4th of 10","society":"Gurukrupa Galaxy","possession":"Ready to Move","rera_id":"P99000005812",
         "price_per_sqft":"Rs. 4,200/sqft","days_on_market":14,"zone":"western",
         "description":"Affordable 1 BHK in Gurukrupa Galaxy, Nalasopara East — great for investors.",
         "features":["modular kitchen","intercom","water storage"],
         "amenities":["watchman","generator","garden"],"source":"demo",
         **_p(2_016_000)},
        {"id":"NS-03","address":"Poonam Gardens Phase 3, Flat 501, Nallasopara West, Mumbai-401203",
         "bedrooms":2,"bathrooms":2,"sqft":920,"carpet_area_sqft":690,"type":"flat","status":"for rent",
         "floor":"5th of 12","society":"Poonam Gardens","possession":"Immediate","rera_id":None,
         "price_per_sqft":"Rs. 11/sqft/month","days_on_market":3,"zone":"western",
         "description":"2 BHK on rent in Poonam Gardens, Nalasopara West — unfurnished, near station.",
         "features":["fan/light fittings","2 bathrooms","covered parking"],
         "amenities":["watchman","CCTV","garden"],"source":"demo",
         **_r(10_000)},
    ],

    # ── VASAI ─────────────────────────────────────────────────────────────────
    "vasai": [
        {"id":"VA-01","address":"Neptune Celestia, Wing B, Flat 702, Vasai West, Mumbai-401202",
         "bedrooms":2,"bathrooms":2,"sqft":960,"carpet_area_sqft":720,"type":"flat","status":"for sale",
         "floor":"7th of 18","society":"Neptune Celestia","possession":"Ready to Move","rera_id":"P99000009876",
         "price_per_sqft":"Rs. 6,500/sqft","days_on_market":17,"zone":"western",
         "description":"2 BHK in Neptune Celestia, Vasai West — premium society near Vasai Road station with creek views.",
         "features":["creek view","modular kitchen","covered parking","power backup"],
         "amenities":["swimming pool","gym","clubhouse","CCTV"],"source":"demo",
         **_p(6_240_000)},
        {"id":"VA-02","address":"Cosmos Residency, Flat 304, Station Road, Vasai Road East, Mumbai-401208",
         "bedrooms":3,"bathrooms":2,"sqft":1150,"carpet_area_sqft":862,"type":"flat","status":"for sale",
         "floor":"3rd of 12","society":"Cosmos Residency","possession":"Ready to Move","rera_id":"P99000008123",
         "price_per_sqft":"Rs. 6,800/sqft","days_on_market":28,"zone":"western",
         "description":"Spacious 3 BHK in Cosmos Residency, Vasai East — walking distance to Vasai Road station.",
         "features":["modular kitchen","2 covered parking","24hr water supply","vastu compliant"],
         "amenities":["gym","children play area","garden"],"source":"demo",
         **_p(7_820_000)},
        {"id":"VA-03","address":"Shree Ganesh CHS, Flat 202, Arnala Road, Vasai West, Mumbai-401201",
         "bedrooms":1,"bathrooms":1,"sqft":520,"carpet_area_sqft":390,"type":"flat","status":"for rent",
         "floor":"2nd of 8","society":"Shree Ganesh CHS","possession":"Immediate","rera_id":None,
         "price_per_sqft":"Rs. 14/sqft/month","days_on_market":6,"zone":"western",
         "description":"1 BHK on rent in Vasai West — near church, quiet lane, semi-furnished.",
         "features":["semi-furnished","fan/light fittings","water storage"],
         "amenities":["watchman","garden"],"source":"demo",
         **_r(7_500)},
    ],

    # ── NAIGAON ───────────────────────────────────────────────────────────────
    "naigaon": [
        {"id":"NG-01","address":"Shree Sai Dham, Wing A, Flat 502, Naigaon West, Mumbai-401207",
         "bedrooms":2,"bathrooms":2,"sqft":900,"carpet_area_sqft":675,"type":"flat","status":"for sale",
         "floor":"5th of 14","society":"Shree Sai Dham","possession":"Ready to Move","rera_id":"P99000007341",
         "price_per_sqft":"Rs. 5,500/sqft","days_on_market":20,"zone":"western",
         "description":"2 BHK in Shree Sai Dham, Naigaon West — gated society 5 min walk to Naigaon station.",
         "features":["modular kitchen","covered parking","power backup","CCTV"],
         "amenities":["gym","garden","children play area"],"source":"demo",
         **_p(4_950_000)},
        {"id":"NG-02","address":"Anand Heights, Flat 301, Pelhar Road, Naigaon East, Mumbai-401208",
         "bedrooms":1,"bathrooms":1,"sqft":550,"carpet_area_sqft":412,"type":"flat","status":"for sale",
         "floor":"3rd of 10","society":"Anand Heights","possession":"Ready to Move","rera_id":"P99000006234",
         "price_per_sqft":"Rs. 4,800/sqft","days_on_market":12,"zone":"western",
         "description":"Budget 1 BHK in Naigaon East — affordable entry-level flat near station.",
         "features":["modular kitchen","water storage","intercom"],
         "amenities":["watchman","garden"],"source":"demo",
         **_p(2_640_000)},
    ],

    # ── BHAYANDAR / BHAYANDER ─────────────────────────────────────────────────
    "bhayander": [
        {"id":"BH-01","address":"Sunteck West World, Tower 1, Flat 1104, Bhayandar West, Thane-401101",
         "bedrooms":2,"bathrooms":2,"sqft":950,"carpet_area_sqft":714,"type":"flat","status":"for sale",
         "floor":"11th of 30","society":"Sunteck West World","possession":"Ready to Move","rera_id":"P51700025678",
         "price_per_sqft":"Rs. 8,200/sqft","days_on_market":16,"zone":"western",
         "description":"2 BHK in Sunteck West World, Bhayandar West — landmark high-rise with Vasai Creek views and premium amenities.",
         "features":["creek view","modular kitchen","2 covered parking","EV charging","smart home"],
         "amenities":["infinity pool","clubhouse","gymnasium","concierge","CCTV"],"source":"demo",
         **_p(7_790_000)},
        {"id":"BH-02","address":"Rustomjee Virar Avenue, Wing C, Flat 802, Bhayandar East, Thane-401105",
         "bedrooms":3,"bathrooms":2,"sqft":1180,"carpet_area_sqft":885,"type":"flat","status":"for sale",
         "floor":"8th of 22","society":"Rustomjee Virar Avenue","possession":"Under Construction — Dec 2025","rera_id":"P51700026543",
         "price_per_sqft":"Rs. 8,800/sqft","days_on_market":35,"zone":"western",
         "description":"3 BHK by Rustomjee in Bhayandar East — integrated township with school, retail & open spaces.",
         "features":["township living","school within complex","modular kitchen","covered parking"],
         "amenities":["swimming pool","gym","retail shops","school","garden"],"source":"demo",
         **_p(10_384_000)},
        {"id":"BH-03","address":"Evershine Millennium Paradise, Flat 504, Golden Nest Road, Bhayandar West, Thane-401101",
         "bedrooms":2,"bathrooms":2,"sqft":880,"carpet_area_sqft":660,"type":"flat","status":"for rent",
         "floor":"5th of 18","society":"Evershine Millennium Paradise","possession":"Immediate","rera_id":None,
         "price_per_sqft":"Rs. 20/sqft/month","days_on_market":7,"zone":"western",
         "description":"2 BHK on rent in Evershine Millennium Paradise, Bhayandar West — semi-furnished, society with pool.",
         "features":["semi-furnished","AC in living room","modular kitchen","1 parking"],
         "amenities":["swimming pool","gym","CCTV","24hr security"],"source":"demo",
         **_r(17_500)},
    ],

    # ── MIRA ROAD ─────────────────────────────────────────────────────────────
    "mira road": [
        {"id":"MR-01","address":"Rustomjee Urbania, La Classique, Wing A, Flat 904, Mira Road East, Thane-401107",
         "bedrooms":2,"bathrooms":2,"sqft":1020,"carpet_area_sqft":765,"type":"flat","status":"for sale",
         "floor":"9th of 25","society":"Rustomjee Urbania — La Classique","possession":"Ready to Move","rera_id":"P51700012345",
         "price_per_sqft":"Rs. 10,500/sqft","days_on_market":14,"zone":"western",
         "description":"2 BHK in Rustomjee Urbania's La Classique, Mira Road East — integrated township with retail, schools & recreation. 3 min to Mira Road station.",
         "features":["modular kitchen","2 covered parking","EV charging","smart locks","vastu compliant"],
         "amenities":["infinity pool","spa","gymnasium","kids zone","retail street","food court"],"source":"demo",
         **_p(10_710_000)},
        {"id":"MR-02","address":"Sunteck West World Tower 3, Flat 1502, Mira Road East, Thane-401107",
         "bedrooms":3,"bathrooms":3,"sqft":1380,"carpet_area_sqft":1035,"type":"flat","status":"for sale",
         "floor":"15th of 32","society":"Sunteck West World","possession":"Ready to Move","rera_id":"P51700013456",
         "price_per_sqft":"Rs. 11,200/sqft","days_on_market":22,"zone":"western",
         "description":"3 BHK in Sunteck West World, Mira Road — panoramic views of Vasai Creek, luxury amenities, near Mira Road station.",
         "features":["creek view","Italian marble floors","2 parking","home automation","servant quarters"],
         "amenities":["rooftop pool","clubhouse","concierge","business centre","gymnasium"],"source":"demo",
         **_p(15_456_000)},
        {"id":"MR-03","address":"Sheth Avalon, Tower B, Flat 602, Beverly Park, Mira Road East, Thane-401107",
         "bedrooms":1,"bathrooms":1,"sqft":600,"carpet_area_sqft":450,"type":"flat","status":"for sale",
         "floor":"6th of 20","society":"Sheth Avalon","possession":"Ready to Move","rera_id":"P51700014567",
         "price_per_sqft":"Rs. 9,800/sqft","days_on_market":8,"zone":"western",
         "description":"1 BHK by Sheth Creators in Avalon, Mira Road — ideal for investment, walkable to station.",
         "features":["modular kitchen","covered parking","power backup","CCTV"],
         "amenities":["gym","garden","children play area","24hr security"],"source":"demo",
         **_p(5_880_000)},
        {"id":"MR-04","address":"DB Orchid Woods, Wing C, Flat 403, Mira Road East, Thane-401107",
         "bedrooms":2,"bathrooms":2,"sqft":950,"carpet_area_sqft":712,"type":"flat","status":"for rent",
         "floor":"4th of 18","society":"DB Orchid Woods","possession":"Immediate","rera_id":None,
         "price_per_sqft":"Rs. 25/sqft/month","days_on_market":5,"zone":"western",
         "description":"Semi-furnished 2 BHK on rent in DB Orchid Woods, Mira Road — AC in all rooms, 10 min walk to Mira Road station.",
         "features":["semi-furnished","AC in all rooms","modular kitchen","1 covered parking"],
         "amenities":["swimming pool","gym","CCTV","24hr security","intercom"],"source":"demo",
         **_r(23_000)},
        {"id":"MR-05","address":"Acme Avenue, Tower A, Flat 1201, Mira Road East, Thane-401107",
         "bedrooms":3,"bathrooms":2,"sqft":1240,"carpet_area_sqft":930,"type":"flat","status":"for rent",
         "floor":"12th of 22","society":"Acme Avenue","possession":"Immediate","rera_id":None,
         "price_per_sqft":"Rs. 28/sqft/month","days_on_market":10,"zone":"western",
         "description":"3 BHK on rent in Acme Avenue, Mira Road — furnished, high floor with open views, near D-Mart.",
         "features":["furnished","AC in all rooms","modular kitchen","2 parking","washing machine"],
         "amenities":["rooftop terrace","gym","CCTV","24hr security"],"source":"demo",
         **_r(34_500)},
    ],

    # ── DAHISAR ───────────────────────────────────────────────────────────────
    "dahisar": [
        {"id":"DH-01","address":"Godrej Alive, Tower A, Flat 1103, Dahisar East, Mumbai-400068",
         "bedrooms":2,"bathrooms":2,"sqft":1020,"carpet_area_sqft":765,"type":"flat","status":"for sale",
         "floor":"11th of 27","society":"Godrej Alive","possession":"Ready to Move","rera_id":"P51800023456",
         "price_per_sqft":"Rs. 12,500/sqft","days_on_market":19,"zone":"western",
         "description":"2 BHK in Godrej Alive, Dahisar East — 32-acre township, Sanjay Gandhi National Park views, near Dahisar metro station.",
         "features":["national park view","modular kitchen","EV charging","covered parking","smart home"],
         "amenities":["swimming pool","clubhouse","gym","jogging track","mini theatre"],"source":"demo",
         **_p(12_750_000)},
        {"id":"DH-02","address":"Runwal Forests, Tower T1, Flat 704, Dahisar East, Mumbai-400068",
         "bedrooms":3,"bathrooms":2,"sqft":1180,"carpet_area_sqft":885,"type":"flat","status":"for sale",
         "floor":"7th of 30","society":"Runwal Forests","possession":"Ready to Move","rera_id":"P51800024567",
         "price_per_sqft":"Rs. 13,200/sqft","days_on_market":27,"zone":"western",
         "description":"3 BHK in Runwal Forests, Dahisar — 11 towers, lush green township bordering Sanjay Gandhi National Park.",
         "features":["forest view","modular kitchen","2 parking","power backup","vastu compliant"],
         "amenities":["multiple pools","forest walking trail","gym","amphitheatre","mini mart"],"source":"demo",
         **_p(15_576_000)},
        {"id":"DH-03","address":"Ekta Tripolis, Wing B, Flat 504, Dahisar West, Mumbai-400091",
         "bedrooms":2,"bathrooms":2,"sqft":950,"carpet_area_sqft":712,"type":"flat","status":"for rent",
         "floor":"5th of 16","society":"Ekta Tripolis","possession":"Immediate","rera_id":None,
         "price_per_sqft":"Rs. 28/sqft/month","days_on_market":6,"zone":"western",
         "description":"2 BHK on rent in Ekta Tripolis, Dahisar West — semi-furnished, near Dahisar check naka.",
         "features":["semi-furnished","AC in bedrooms","modular kitchen","1 parking"],
         "amenities":["gym","CCTV","24hr security","garden"],"source":"demo",
         **_r(26_500)},
    ],

    # ── BORIVALI ──────────────────────────────────────────────────────────────
    "borivali": [
        {"id":"BO-01","address":"Oberoi Eternia, Wing E, Flat 1504, Borivali East, Mumbai-400066",
         "bedrooms":2,"bathrooms":2,"sqft":1105,"carpet_area_sqft":829,"type":"flat","status":"for sale",
         "floor":"15th of 30","society":"Oberoi Eternia","possession":"Ready to Move","rera_id":"P51800019876",
         "price_per_sqft":"Rs. 18,500/sqft","days_on_market":21,"zone":"western",
         "description":"2 BHK in Oberoi Eternia, Borivali East — premium Oberoi township with Sanjay Gandhi National Park views and world-class amenities.",
         "features":["national park view","Italian marble","modular kitchen","2 parking","smart home"],
         "amenities":["olympic pool","spa","squash court","business lounge","concierge"],"source":"demo",
         **_p(20_442_500)},
        {"id":"BO-02","address":"Dosti Desire, Tower D, Flat 804, Borivali West, Mumbai-400092",
         "bedrooms":2,"bathrooms":2,"sqft":985,"carpet_area_sqft":739,"type":"flat","status":"for sale",
         "floor":"8th of 22","society":"Dosti Desire","possession":"Ready to Move","rera_id":"P51800020123",
         "price_per_sqft":"Rs. 15,800/sqft","days_on_market":16,"zone":"western",
         "description":"2 BHK in Dosti Desire, Borivali West — well-connected to Borivali station and Western Express Highway.",
         "features":["modular kitchen","covered parking","power backup","vastu compliant"],
         "amenities":["rooftop pool","gym","children's play area","CCTV"],"source":"demo",
         **_p(15_563_000)},
        {"id":"BO-03","address":"Lodha Bellagio, Tower A, Flat 1205, Borivali East, Mumbai-400066",
         "bedrooms":3,"bathrooms":3,"sqft":1520,"carpet_area_sqft":1140,"type":"flat","status":"for sale",
         "floor":"12th of 28","society":"Lodha Bellagio","possession":"Ready to Move","rera_id":"P51800020456",
         "price_per_sqft":"Rs. 19,200/sqft","days_on_market":38,"zone":"western",
         "description":"3 BHK in Lodha Bellagio, Borivali East — lifestyle township with retail, entertainment and national park proximity.",
         "features":["national park view","premium fittings","2 parking","servant quarters","EV charging"],
         "amenities":["infinity pool","spa","mini cinema","business centre","retail plaza"],"source":"demo",
         **_p(29_184_000)},
        {"id":"BO-04","address":"Mahavir Darshan CHS, Flat 301, MG Road, Borivali West, Mumbai-400092",
         "bedrooms":2,"bathrooms":2,"sqft":900,"carpet_area_sqft":675,"type":"flat","status":"for rent",
         "floor":"3rd of 10","society":"Mahavir Darshan CHS","possession":"Immediate","rera_id":None,
         "price_per_sqft":"Rs. 33/sqft/month","days_on_market":4,"zone":"western",
         "description":"2 BHK on rent in Borivali West — 5 min walk to Borivali station, semi-furnished.",
         "features":["semi-furnished","AC in living room","modular kitchen","1 parking"],
         "amenities":["watchman","CCTV","garden"],"source":"demo",
         **_r(30_000)},
    ],

    # ── KANDIVALI ─────────────────────────────────────────────────────────────
    "kandivali": [
        {"id":"KD-01","address":"Mahindra Happinest Kalyan, Wing 1, Flat 603, Kandivali East, Mumbai-400101",
         "bedrooms":1,"bathrooms":1,"sqft":672,"carpet_area_sqft":504,"type":"flat","status":"for sale",
         "floor":"6th of 18","society":"Mahindra Happinest","possession":"Ready to Move","rera_id":"P51800016543",
         "price_per_sqft":"Rs. 14,200/sqft","days_on_market":12,"zone":"western",
         "description":"1 BHK in Mahindra Happinest, Kandivali East — smart-home enabled, near metro station.",
         "features":["smart home","modular kitchen","covered parking","solar lighting"],
         "amenities":["gym","garden","children play area","CCTV"],"source":"demo",
         **_p(9_542_400)},
        {"id":"KD-02","address":"Godrej Tranquil, Tower A, Flat 1102, Kandivali East, Mumbai-400101",
         "bedrooms":2,"bathrooms":2,"sqft":1050,"carpet_area_sqft":788,"type":"flat","status":"for sale",
         "floor":"11th of 24","society":"Godrej Tranquil","possession":"Ready to Move","rera_id":"P51800017234",
         "price_per_sqft":"Rs. 16,800/sqft","days_on_market":24,"zone":"western",
         "description":"2 BHK in Godrej Tranquil, Kandivali East — close to Metro Line 7 (Kandivali station).",
         "features":["metro proximity","modular kitchen","EV charging","covered parking"],
         "amenities":["swimming pool","gym","co-working space","CCTV"],"source":"demo",
         **_p(17_640_000)},
        {"id":"KD-03","address":"Runwal Forests, Tower R2, Flat 905, Kandivali East, Mumbai-400101",
         "bedrooms":3,"bathrooms":2,"sqft":1240,"carpet_area_sqft":930,"type":"flat","status":"for sale",
         "floor":"9th of 26","society":"Runwal Forests","possession":"Ready to Move","rera_id":"P51800017890",
         "price_per_sqft":"Rs. 17,500/sqft","days_on_market":31,"zone":"western",
         "description":"3 BHK in Runwal Forests, Kandivali East — 25 acres of greenery, Sanjay Gandhi National Park bordering.",
         "features":["green view","2 parking","modular kitchen","power backup"],
         "amenities":["forest walk","multiple pools","amphitheatre","gym"],"source":"demo",
         **_p(21_700_000)},
        {"id":"KD-04","address":"Evershine Crown, Flat 602, SV Road, Kandivali West, Mumbai-400067",
         "bedrooms":2,"bathrooms":2,"sqft":920,"carpet_area_sqft":690,"type":"flat","status":"for rent",
         "floor":"6th of 14","society":"Evershine Crown","possession":"Immediate","rera_id":None,
         "price_per_sqft":"Rs. 32/sqft/month","days_on_market":8,"zone":"western",
         "description":"2 BHK on rent in Evershine Crown, Kandivali West — near Kandivali station & Infiniti Mall.",
         "features":["semi-furnished","AC in bedrooms","modular kitchen","1 parking"],
         "amenities":["gym","swimming pool","CCTV","24hr security"],"source":"demo",
         **_r(29_500)},
    ],

    # ── MALAD ─────────────────────────────────────────────────────────────────
    "malad": [
        {"id":"ML-01","address":"L&T Elixir Reserve, Tower A, Flat 2104, Malad West, Mumbai-400064",
         "bedrooms":2,"bathrooms":2,"sqft":1100,"carpet_area_sqft":825,"type":"flat","status":"for sale",
         "floor":"21st of 35","society":"L&T Elixir Reserve","possession":"Ready to Move","rera_id":"P51800013456",
         "price_per_sqft":"Rs. 21,500/sqft","days_on_market":18,"zone":"western",
         "description":"2 BHK in L&T Elixir Reserve, Malad West — high-rise with sea views, near Mindspace IT Park and Metro Line 2A.",
         "features":["partial sea view","modular kitchen","EV charging","smart home","2 parking"],
         "amenities":["infinity pool","co-working lounge","gym","sky deck","CCTV"],"source":"demo",
         **_p(23_650_000)},
        {"id":"ML-02","address":"Kalpataru Aura, Wing B, Flat 1504, Malad West, Mumbai-400064",
         "bedrooms":3,"bathrooms":3,"sqft":1490,"carpet_area_sqft":1118,"type":"flat","status":"for sale",
         "floor":"15th of 32","society":"Kalpataru Aura","possession":"Ready to Move","rera_id":"P51800014123",
         "price_per_sqft":"Rs. 23,000/sqft","days_on_market":29,"zone":"western",
         "description":"3 BHK in Kalpataru Aura, Malad West — premium high-rise with sea-facing views, near Malad station.",
         "features":["sea view","Italian marble","2 parking","servant quarters","power backup"],
         "amenities":["rooftop pool","spa","gym","concierge","children's play area"],"source":"demo",
         **_p(34_270_000)},
        {"id":"ML-03","address":"Indiabulls Blu Estate, Flat 804, Marve Road, Malad West, Mumbai-400064",
         "bedrooms":2,"bathrooms":2,"sqft":1050,"carpet_area_sqft":788,"type":"flat","status":"for rent",
         "floor":"8th of 24","society":"Indiabulls Blu Estate","possession":"Immediate","rera_id":None,
         "price_per_sqft":"Rs. 42/sqft/month","days_on_market":6,"zone":"western",
         "description":"Semi-furnished 2 BHK on rent in Indiabulls Blu Estate, Malad West — gated complex, near Malad metro.",
         "features":["semi-furnished","AC in all rooms","modular kitchen","1 parking"],
         "amenities":["swimming pool","gym","24hr security","CCTV"],"source":"demo",
         **_r(44_000)},
    ],

    # ── GOREGAON ──────────────────────────────────────────────────────────────
    "goregaon": [
        {"id":"GG-01","address":"Oberoi Exquisite, Wing C, Flat 1803, Goregaon East, Mumbai-400063",
         "bedrooms":3,"bathrooms":3,"sqft":1520,"carpet_area_sqft":1140,"type":"flat","status":"for sale",
         "floor":"18th of 40","society":"Oberoi Exquisite","possession":"Ready to Move","rera_id":"P51800011234",
         "price_per_sqft":"Rs. 22,500/sqft","days_on_market":35,"zone":"western",
         "description":"3 BHK in Oberoi Exquisite, Goregaon East — Oberoi's landmark high-rise, proximity to Aarey Colony, Film City.",
         "features":["panoramic view","Italian marble","2 parking","smart home","wine cellar"],
         "amenities":["rooftop pool","spa","home theatre","concierge","private cinema"],"source":"demo",
         **_p(34_200_000)},
        {"id":"GG-02","address":"Lodha Prime Square, Tower B, Flat 1204, Goregaon West, Mumbai-400062",
         "bedrooms":2,"bathrooms":2,"sqft":1080,"carpet_area_sqft":810,"type":"flat","status":"for sale",
         "floor":"12th of 28","society":"Lodha Prime Square","possession":"Ready to Move","rera_id":"P51800012345",
         "price_per_sqft":"Rs. 20,000/sqft","days_on_market":22,"zone":"western",
         "description":"2 BHK in Lodha Prime Square, Goregaon West — near Goregaon station and Western Express Highway.",
         "features":["modular kitchen","2 parking","EV charging","covered parking"],
         "amenities":["swimming pool","gym","clubhouse","indoor games"],"source":"demo",
         **_p(21_600_000)},
        {"id":"GG-03","address":"RNA NG Royal Park, Wing A, Flat 702, Goregaon West, Mumbai-400062",
         "bedrooms":2,"bathrooms":2,"sqft":980,"carpet_area_sqft":735,"type":"flat","status":"for rent",
         "floor":"7th of 20","society":"RNA NG Royal Park","possession":"Immediate","rera_id":None,
         "price_per_sqft":"Rs. 42/sqft/month","days_on_market":9,"zone":"western",
         "description":"2 BHK on rent in RNA NG Royal Park, Goregaon West — near Goregaon station, semi-furnished.",
         "features":["semi-furnished","AC in bedrooms","modular kitchen","1 parking"],
         "amenities":["gym","children play area","CCTV","24hr security"],"source":"demo",
         **_r(41_000)},
    ],

    # ── ANDHERI ───────────────────────────────────────────────────────────────
    "andheri": [
        {"id":"AN-01","address":"Rustomjee Elanza, Wing A, Flat 1404, Andheri West, Mumbai-400053",
         "bedrooms":2,"bathrooms":2,"sqft":1180,"carpet_area_sqft":885,"type":"flat","status":"for sale",
         "floor":"14th of 30","society":"Rustomjee Elanza","possession":"Ready to Move","rera_id":"P51800007654",
         "price_per_sqft":"Rs. 26,000/sqft","days_on_market":17,"zone":"western",
         "description":"2 BHK in Rustomjee Elanza, Andheri West — luxury high-rise near Versova Metro, 10 min to airport.",
         "features":["city view","modular kitchen","2 parking","EV charging","smart home"],
         "amenities":["infinity pool","spa","co-working space","gymnasium","sky lounge"],"source":"demo",
         **_p(30_680_000)},
        {"id":"AN-02","address":"Sunteck Signia Isles, Tower 1, Flat 2204, Andheri East, Mumbai-400059",
         "bedrooms":3,"bathrooms":3,"sqft":1650,"carpet_area_sqft":1238,"type":"flat","status":"for sale",
         "floor":"22nd of 40","society":"Sunteck Signia Isles","possession":"Ready to Move","rera_id":"P51800008765",
         "price_per_sqft":"Rs. 25,000/sqft","days_on_market":42,"zone":"western",
         "description":"3 BHK in Sunteck Signia Isles, Andheri East — BKC proximity, JVLR connectivity, Metro Line 1.",
         "features":["skyline view","Italian marble","2 parking","servant quarters","power backup"],
         "amenities":["rooftop pool","concierge","gymnasium","home theatre","valet parking"],"source":"demo",
         **_p(41_250_000)},
        {"id":"AN-03","address":"Kalpataru Magnus, Wing B, Flat 904, Andheri East, Mumbai-400069",
         "bedrooms":2,"bathrooms":2,"sqft":1100,"carpet_area_sqft":825,"type":"flat","status":"for rent",
         "floor":"9th of 22","society":"Kalpataru Magnus","possession":"Immediate","rera_id":None,
         "price_per_sqft":"Rs. 57/sqft/month","days_on_market":7,"zone":"western",
         "description":"2 BHK on rent in Kalpataru Magnus, Andheri East — near Metro Line 1, SEEPZ, and JVLR.",
         "features":["semi-furnished","AC in all rooms","modular kitchen","1 parking"],
         "amenities":["swimming pool","gym","CCTV","24hr security","indoor games"],"source":"demo",
         **_r(62_000)},
        {"id":"AN-04","address":"Seven Eleven CHS, Flat 401, Versova Road, Andheri West, Mumbai-400061",
         "bedrooms":1,"bathrooms":1,"sqft":620,"carpet_area_sqft":465,"type":"flat","status":"for rent",
         "floor":"4th of 8","society":"Seven Eleven CHS","possession":"Immediate","rera_id":None,
         "price_per_sqft":"Rs. 55/sqft/month","days_on_market":3,"zone":"western",
         "description":"1 BHK on rent in Andheri West near Versova Metro — unfurnished, walking distance to station.",
         "features":["fan/light fittings","attached bathroom","open kitchen"],
         "amenities":["watchman","garden"],"source":"demo",
         **_r(34_000)},
    ],

    # ── JUHU ──────────────────────────────────────────────────────────────────
    "juhu": [
        {"id":"JU-01","address":"JVPD Scheme, Bungalow 14, Juhu Tara Road, Juhu, Mumbai-400049",
         "bedrooms":4,"bathrooms":4,"sqft":4200,"carpet_area_sqft":3150,"type":"villa","status":"for sale",
         "floor":"Ground","society":"JVPD Scheme","possession":"Ready to Move","rera_id":"P51800006543",
         "price_per_sqft":"Rs. 65,000/sqft","days_on_market":60,"zone":"western",
         "description":"Independent 4 BHK bungalow in JVPD Scheme, Juhu — 15 min walk to Juhu Beach, gated society, private garden.",
         "features":["private garden","sea proximity","covered garage","servant quarters","terrace"],
         "amenities":["private pool","private garden","3-car garage"],"source":"demo",
         **_p(273_000_000)},
        {"id":"JU-02","address":"Sea Queen CHS, Flat 602, Juhu Versova Link Road, Juhu, Mumbai-400049",
         "bedrooms":3,"bathrooms":3,"sqft":1800,"carpet_area_sqft":1350,"type":"flat","status":"for sale",
         "floor":"6th of 10","society":"Sea Queen CHS","possession":"Ready to Move","rera_id":"P51800007123",
         "price_per_sqft":"Rs. 55,000/sqft","days_on_market":45,"zone":"western",
         "description":"3 BHK in Sea Queen CHS, Juhu — sea-facing, steps from Juhu Beach, low-rise boutique building.",
         "features":["sea view","private terrace","2 parking","split ACs","modular kitchen"],
         "amenities":["watchman","generator","CCTV"],"source":"demo",
         **_p(99_000_000)},
        {"id":"JU-03","address":"Goldline Apartments, Flat 304, Juhu Church Road, Juhu, Mumbai-400049",
         "bedrooms":2,"bathrooms":2,"sqft":1200,"carpet_area_sqft":900,"type":"flat","status":"for rent",
         "floor":"3rd of 8","society":"Goldline Apartments","possession":"Immediate","rera_id":None,
         "price_per_sqft":"Rs. 115/sqft/month","days_on_market":5,"zone":"western",
         "description":"2 BHK on rent in Juhu — fully furnished, 5 min walk to Juhu Beach, near JVPD market.",
         "features":["fully furnished","AC in all rooms","modular kitchen","1 parking"],
         "amenities":["watchman","generator","CCTV"],"source":"demo",
         **_r(138_000)},
    ],

    # ── BANDRA ────────────────────────────────────────────────────────────────
    "bandra": [
        {"id":"BA-01","address":"Kalpataru Avana, Wing A, Flat 2104, Bandra West, Mumbai-400050",
         "bedrooms":3,"bathrooms":3,"sqft":1850,"carpet_area_sqft":1388,"type":"flat","status":"for sale",
         "floor":"21st of 36","society":"Kalpataru Avana","possession":"Ready to Move","rera_id":"P51800003456",
         "price_per_sqft":"Rs. 55,000/sqft","days_on_market":38,"zone":"western",
         "description":"3 BHK in Kalpataru Avana, Bandra West — sea-facing luxury, steps from Bandstand, BKC 10 min.",
         "features":["sea view","Italian marble","2 parking","home automation","butler service"],
         "amenities":["infinity pool","spa","gym","concierge","valet parking"],"source":"demo",
         **_p(101_750_000)},
        {"id":"BA-02","address":"Rustomjee Crown, Wing B, Flat 1504, Bandra West, Mumbai-400050",
         "bedrooms":2,"bathrooms":2,"sqft":1350,"carpet_area_sqft":1013,"type":"flat","status":"for sale",
         "floor":"15th of 30","society":"Rustomjee Crown","possession":"Ready to Move","rera_id":"P51800004123",
         "price_per_sqft":"Rs. 50,000/sqft","days_on_market":25,"zone":"western",
         "description":"2 BHK in Rustomjee Crown, Bandra West — city-facing luxury, close to Linking Road and St. Andrew's.",
         "features":["city view","premium fittings","2 parking","smart home","modular kitchen"],
         "amenities":["rooftop pool","sky lounge","gym","co-working","CCTV"],"source":"demo",
         **_p(67_500_000)},
        {"id":"BA-03","address":"Mount Mary Road, Flat 401, Bandra West, Mumbai-400050",
         "bedrooms":2,"bathrooms":2,"sqft":1100,"carpet_area_sqft":825,"type":"flat","status":"for rent",
         "floor":"4th of 8","society":"Rosemary Apartments","possession":"Immediate","rera_id":None,
         "price_per_sqft":"Rs. 120/sqft/month","days_on_market":4,"zone":"western",
         "description":"2 BHK on rent on Mount Mary Road, Bandra West — fully furnished, near Bandstand and St. Andrew's church.",
         "features":["fully furnished","AC in all rooms","modular kitchen","1 parking"],
         "amenities":["watchman","generator","CCTV"],"source":"demo",
         **_r(132_000)},
        {"id":"BA-04","address":"Godrej Sky Garden, Flat 604, Bandra East, Mumbai-400051",
         "bedrooms":2,"bathrooms":2,"sqft":1020,"carpet_area_sqft":765,"type":"flat","status":"for sale",
         "floor":"6th of 22","society":"Godrej Sky Garden","possession":"Ready to Move","rera_id":"P51800004567",
         "price_per_sqft":"Rs. 38,000/sqft","days_on_market":21,"zone":"western",
         "description":"2 BHK in Godrej Sky Garden, Bandra East — BKC-adjacent, metro Line 3 station close by.",
         "features":["BKC proximity","modular kitchen","covered parking","power backup"],
         "amenities":["swimming pool","gym","children play area","CCTV"],"source":"demo",
         **_p(38_760_000)},
    ],

    # ── LOWER PAREL ───────────────────────────────────────────────────────────
    "lower parel": [
        {"id":"LP-01","address":"Lodha Park, Tower A, Flat 2804, Lower Parel, Mumbai-400013",
         "bedrooms":3,"bathrooms":3,"sqft":2100,"carpet_area_sqft":1575,"type":"flat","status":"for sale",
         "floor":"28th of 55","society":"Lodha Park","possession":"Ready to Move","rera_id":"P51800001234",
         "price_per_sqft":"Rs. 65,000/sqft","days_on_market":52,"zone":"south",
         "description":"3 BHK in Lodha Park, Lower Parel — ultra-luxury with private pool terrace, steps from Palladium Mall.",
         "features":["city skyline view","private pool terrace","Italian marble","butler service","wine cellar"],
         "amenities":["infinity sky pool","private cinema","concierge","valet parking","spa"],"source":"demo",
         **_p(136_500_000)},
        {"id":"LP-02","address":"Piramal Aranya, Tower B, Flat 1604, Byculla, Mumbai-400027",
         "bedrooms":2,"bathrooms":2,"sqft":1380,"carpet_area_sqft":1035,"type":"flat","status":"for sale",
         "floor":"16th of 62","society":"Piramal Aranya","possession":"Ready to Move","rera_id":"P51800002345",
         "price_per_sqft":"Rs. 55,000/sqft","days_on_market":30,"zone":"south",
         "description":"2 BHK in Piramal Aranya — India's tallest green building, 2.5-acre podium garden, city & sea views.",
         "features":["sea view","green LEED certified","modular kitchen","2 parking","EV charging"],
         "amenities":["sky infinity pool","amphitheatre","yoga lawn","gym","concierge"],"source":"demo",
         **_p(75_900_000)},
        {"id":"LP-03","address":"Peninsula Heights, Flat 1204, Ganpatrao Kadam Marg, Lower Parel, Mumbai-400013",
         "bedrooms":3,"bathrooms":3,"sqft":1950,"carpet_area_sqft":1463,"type":"flat","status":"for rent",
         "floor":"12th of 30","society":"Peninsula Heights","possession":"Immediate","rera_id":None,
         "price_per_sqft":"Rs. 135/sqft/month","days_on_market":8,"zone":"south",
         "description":"3 BHK on rent in Peninsula Heights, Lower Parel — furnished, Palladium Mall walkable, near office hubs.",
         "features":["fully furnished","AC in all rooms","modular kitchen","2 parking","washing machine"],
         "amenities":["rooftop pool","gym","concierge","CCTV","valet parking"],"source":"demo",
         **_r(263_000)},
    ],

    # ── WORLI ─────────────────────────────────────────────────────────────────
    "worli": [
        {"id":"WO-01","address":"Three Sixty West, Tower A, Flat 4802, Worli, Mumbai-400018",
         "bedrooms":4,"bathrooms":4,"sqft":4500,"carpet_area_sqft":3375,"type":"penthouse","status":"for sale",
         "floor":"48th of 65","society":"Three Sixty West (Oberoi Realty)","possession":"Ready to Move","rera_id":"P51800000123",
         "price_per_sqft":"Rs. 1,20,000/sqft","days_on_market":90,"zone":"south",
         "description":"4 BHK super-luxury penthouse in Three Sixty West, Worli — Mumbai's most iconic address, 360° sea views, designed by Oberoi Realty with Armani/Casa interiors.",
         "features":["360° sea view","Armani/Casa interiors","private terrace","butler service","2 staff rooms"],
         "amenities":["private pool","Nobu restaurant","concierge","helipad","valet"],"source":"demo",
         **_p(540_000_000)},
        {"id":"WO-02","address":"Raheja Vivarea, Tower C, Flat 2404, Worli, Mumbai-400018",
         "bedrooms":3,"bathrooms":3,"sqft":2600,"carpet_area_sqft":1950,"type":"flat","status":"for sale",
         "floor":"24th of 48","society":"Raheja Vivarea","possession":"Ready to Move","rera_id":"P51800000456",
         "price_per_sqft":"Rs. 75,000/sqft","days_on_market":55,"zone":"south",
         "description":"3 BHK in Raheja Vivarea, Worli — sea-link view, Sea Face promenade access, ultra-premium finishes.",
         "features":["sea link view","marble flooring","2 parking","home automation","servant quarters"],
         "amenities":["infinity pool","spa","private cinema","gym","concierge"],"source":"demo",
         **_p(195_000_000)},
        {"id":"WO-03","address":"Kalpataru Primus, Flat 1604, Worli Seaface, Mumbai-400030",
         "bedrooms":2,"bathrooms":2,"sqft":1650,"carpet_area_sqft":1238,"type":"flat","status":"for rent",
         "floor":"16th of 35","society":"Kalpataru Primus","possession":"Immediate","rera_id":None,
         "price_per_sqft":"Rs. 200/sqft/month","days_on_market":10,"zone":"south",
         "description":"2 BHK on rent in Kalpataru Primus, Worli Sea Face — furnished, direct sea view, near Worli-BKC arterial.",
         "features":["fully furnished","sea view","AC in all rooms","2 parking","modular kitchen"],
         "amenities":["infinity pool","spa","gym","concierge","24hr security"],"source":"demo",
         **_r(330_000)},
    ],

    # ── POWAI ─────────────────────────────────────────────────────────────────
    "powai": [
        {"id":"PW-01","address":"Hiranandani Gardens, Atlantis, Flat 1204, Powai, Mumbai-400076",
         "bedrooms":2,"bathrooms":2,"sqft":1200,"carpet_area_sqft":900,"type":"flat","status":"for sale",
         "floor":"12th of 22","society":"Hiranandani Gardens — Atlantis","possession":"Ready to Move","rera_id":"P51800015678",
         "price_per_sqft":"Rs. 24,000/sqft","days_on_market":28,"zone":"central",
         "description":"2 BHK in Hiranandani Gardens (Atlantis Tower), Powai — lake view, planned township, IIT Mumbai proximity.",
         "features":["lake view","modular kitchen","covered parking","power backup","vastu compliant"],
         "amenities":["multiple pools","clubhouse","jogging track","school nearby","hospital nearby"],"source":"demo",
         **_p(28_800_000)},
        {"id":"PW-02","address":"Lodha Amara, Tower 3, Flat 3604, Kolshet Road, Powai, Mumbai-400076",
         "bedrooms":3,"bathrooms":3,"sqft":1600,"carpet_area_sqft":1200,"type":"flat","status":"for sale",
         "floor":"36th of 50","society":"Lodha Amara","possession":"Ready to Move","rera_id":"P51800016234",
         "price_per_sqft":"Rs. 26,500/sqft","days_on_market":35,"zone":"central",
         "description":"3 BHK in Lodha Amara, Powai — panoramic lake and city views, 10 min to IIT, near Hiranandani Hospital.",
         "features":["lake & city view","Italian marble","2 parking","smart home","EV charging"],
         "amenities":["rooftop pool","spa","amphitheatre","gym","food court"],"source":"demo",
         **_p(42_400_000)},
        {"id":"PW-03","address":"Hiranandani Gardens, Cleo, Flat 804, Powai, Mumbai-400076",
         "bedrooms":2,"bathrooms":2,"sqft":1150,"carpet_area_sqft":863,"type":"flat","status":"for rent",
         "floor":"8th of 18","society":"Hiranandani Gardens — Cleo","possession":"Immediate","rera_id":None,
         "price_per_sqft":"Rs. 62/sqft/month","days_on_market":5,"zone":"central",
         "description":"2 BHK on rent in Hiranandani Gardens (Cleo Tower), Powai — furnished, lake proximity, near NITIE.",
         "features":["semi-furnished","AC in all rooms","modular kitchen","1 parking"],
         "amenities":["swimming pool","gym","garden","24hr security","CCTV"],"source":"demo",
         **_r(71_000)},
    ],

    # ── CHEMBUR ───────────────────────────────────────────────────────────────
    "chembur": [
        {"id":"CH-01","address":"Godrej Nurture, Wing A, Flat 1104, Chembur, Mumbai-400071",
         "bedrooms":2,"bathrooms":2,"sqft":1080,"carpet_area_sqft":810,"type":"flat","status":"for sale",
         "floor":"11th of 24","society":"Godrej Nurture","possession":"Ready to Move","rera_id":"P51800018765",
         "price_per_sqft":"Rs. 22,000/sqft","days_on_market":19,"zone":"central",
         "description":"2 BHK in Godrej Nurture, Chembur — near Chembur Monorail & Metro Line 2B, BKC accessible via Eastern Freeway.",
         "features":["modular kitchen","covered parking","EV charging","smart home","power backup"],
         "amenities":["swimming pool","gym","children play area","co-working space","CCTV"],"source":"demo",
         **_p(23_760_000)},
        {"id":"CH-02","address":"Runwal Bliss, Tower B, Flat 2204, Chembur, Mumbai-400071",
         "bedrooms":3,"bathrooms":3,"sqft":1450,"carpet_area_sqft":1088,"type":"flat","status":"for sale",
         "floor":"22nd of 38","society":"Runwal Bliss","possession":"Ready to Move","rera_id":"P51800019234",
         "price_per_sqft":"Rs. 24,000/sqft","days_on_market":32,"zone":"central",
         "description":"3 BHK in Runwal Bliss, Chembur — Eastern Express Highway frontage, city views, monorail connectivity.",
         "features":["city view","Italian marble","2 parking","servant quarters","EV charging"],
         "amenities":["rooftop pool","gym","amphitheatre","indoor games","concierge"],"source":"demo",
         **_p(34_800_000)},
        {"id":"CH-03","address":"Surana Sethia Empire, Flat 604, Chembur Main Road, Chembur, Mumbai-400071",
         "bedrooms":2,"bathrooms":2,"sqft":1020,"carpet_area_sqft":765,"type":"flat","status":"for rent",
         "floor":"6th of 16","society":"Surana Sethia Empire","possession":"Immediate","rera_id":None,
         "price_per_sqft":"Rs. 48/sqft/month","days_on_market":7,"zone":"central",
         "description":"2 BHK on rent in Surana Sethia Empire, Chembur — near RCF colony, Monorail and Chembur station.",
         "features":["semi-furnished","AC in living room","modular kitchen","1 parking"],
         "amenities":["gym","garden","CCTV","24hr security"],"source":"demo",
         **_r(49_000)},
    ],

    # ── DADAR ─────────────────────────────────────────────────────────────────
    "dadar": [
        {"id":"DD-01","address":"Ruparel Orion, Wing A, Flat 1504, Dadar West, Mumbai-400028",
         "bedrooms":3,"bathrooms":3,"sqft":1680,"carpet_area_sqft":1260,"type":"flat","status":"for sale",
         "floor":"15th of 30","society":"Ruparel Orion","possession":"Ready to Move","rera_id":"P51800009876",
         "price_per_sqft":"Rs. 38,000/sqft","days_on_market":33,"zone":"central",
         "description":"3 BHK in Ruparel Orion, Dadar West — walking distance to Dadar station (Central+Western), near Shivaji Park.",
         "features":["city view","modular kitchen","2 parking","smart home","power backup"],
         "amenities":["rooftop pool","gym","co-working space","indoor games","CCTV"],"source":"demo",
         **_p(63_840_000)},
        {"id":"DD-02","address":"Piramal Vaikunth, Tower B, Flat 804, Dadar East, Mumbai-400014",
         "bedrooms":2,"bathrooms":2,"sqft":1100,"carpet_area_sqft":825,"type":"flat","status":"for sale",
         "floor":"8th of 22","society":"Piramal Vaikunth","possession":"Ready to Move","rera_id":"P51800010234",
         "price_per_sqft":"Rs. 32,000/sqft","days_on_market":26,"zone":"central",
         "description":"2 BHK in Piramal Vaikunth, Dadar East — at the crossroads of Central and Western lines, superb connectivity.",
         "features":["modular kitchen","covered parking","power backup","vastu compliant"],
         "amenities":["swimming pool","gym","clubhouse","CCTV"],"source":"demo",
         **_p(35_200_000)},
        {"id":"DD-03","address":"Shivaji Park CHS, Flat 302, Veer Savarkar Marg, Dadar West, Mumbai-400028",
         "bedrooms":2,"bathrooms":1,"sqft":920,"carpet_area_sqft":690,"type":"flat","status":"for rent",
         "floor":"3rd of 7","society":"Shivaji Park CHS","possession":"Immediate","rera_id":None,
         "price_per_sqft":"Rs. 88/sqft/month","days_on_market":5,"zone":"central",
         "description":"2 BHK on rent facing Shivaji Park, Dadar West — unfurnished, old building with great location.",
         "features":["park view","fan/light fittings","open kitchen","cross ventilation"],
         "amenities":["watchman"],"source":"demo",
         **_r(81_000)},
    ],

    # ── WADALA ────────────────────────────────────────────────────────────────
    "wadala": [
        {"id":"WD-01","address":"Wadhwa The Address by GS, Tower B, Flat 1504, Wadala, Mumbai-400037",
         "bedrooms":2,"bathrooms":2,"sqft":1120,"carpet_area_sqft":840,"type":"flat","status":"for sale",
         "floor":"15th of 32","society":"Wadhwa The Address by GS","possession":"Ready to Move","rera_id":"P51800022345",
         "price_per_sqft":"Rs. 22,500/sqft","days_on_market":22,"zone":"harbour",
         "description":"2 BHK in Wadhwa The Address, Wadala — monorail at doorstep, Eastern Freeway access, BKC 15 min.",
         "features":["city view","modular kitchen","EV charging","covered parking","smart home"],
         "amenities":["rooftop pool","gym","co-working space","CCTV","children play area"],"source":"demo",
         **_p(25_200_000)},
        {"id":"WD-02","address":"Kanakia Rainforest, Wing A, Flat 1004, Wadala, Mumbai-400037",
         "bedrooms":3,"bathrooms":3,"sqft":1480,"carpet_area_sqft":1110,"type":"flat","status":"for sale",
         "floor":"10th of 28","society":"Kanakia Rainforest","possession":"Ready to Move","rera_id":"P51800023123",
         "price_per_sqft":"Rs. 24,000/sqft","days_on_market":38,"zone":"harbour",
         "description":"3 BHK in Kanakia Rainforest, Wadala — green living, near Sewri Mumbai Port Trust land development corridor.",
         "features":["green lungs view","modular kitchen","2 parking","power backup"],
         "amenities":["forest garden","swimming pool","gym","amphitheatre","indoor games"],"source":"demo",
         **_p(35_520_000)},
        {"id":"WD-03","address":"Rajesh Lifespaces White City, Flat 604, Antophill, Wadala East, Mumbai-400037",
         "bedrooms":2,"bathrooms":2,"sqft":1050,"carpet_area_sqft":788,"type":"flat","status":"for rent",
         "floor":"6th of 20","society":"Rajesh White City","possession":"Immediate","rera_id":None,
         "price_per_sqft":"Rs. 45/sqft/month","days_on_market":6,"zone":"harbour",
         "description":"2 BHK on rent in Rajesh White City, Wadala East — near Monorail, Antophill, Eastern Freeway.",
         "features":["semi-furnished","AC in bedrooms","modular kitchen","1 parking"],
         "amenities":["gym","garden","CCTV","24hr security"],"source":"demo",
         **_r(47_000)},
    ],

    # ── COLABA ────────────────────────────────────────────────────────────────
    "colaba": [
        {"id":"CO-01","address":"Cusrow Baug, Flat 501, Shahid Bhagat Singh Road, Colaba, Mumbai-400005",
         "bedrooms":3,"bathrooms":3,"sqft":2200,"carpet_area_sqft":1650,"type":"flat","status":"for sale",
         "floor":"5th of 9","society":"Cusrow Baug","possession":"Ready to Move","rera_id":"P51800000789",
         "price_per_sqft":"Rs. 75,000/sqft","days_on_market":70,"zone":"south",
         "description":"3 BHK in Cusrow Baug, Colaba — iconic heritage colony, steps from Gateway of India, unique art deco structure.",
         "features":["heritage building","sea proximity","high ceilings","large rooms","private garden"],
         "amenities":["watchman","garden","heritage compound"],"source":"demo",
         **_p(165_000_000)},
        {"id":"CO-02","address":"Taj Mahal Tower, Flat 1201, Apollo Bunder, Colaba, Mumbai-400001",
         "bedrooms":2,"bathrooms":2,"sqft":1600,"carpet_area_sqft":1200,"type":"flat","status":"for rent",
         "floor":"12th of 24","society":"Taj Mahal Tower Residences","possession":"Immediate","rera_id":None,
         "price_per_sqft":"Rs. 200/sqft/month","days_on_market":14,"zone":"south",
         "description":"2 BHK on rent at Colaba — harbour view, Gateway of India proximity, corporate lease preferred.",
         "features":["harbour view","fully furnished","concierge","2 parking","smart home"],
         "amenities":["pool","gym","concierge","valet","24hr security"],"source":"demo",
         **_r(320_000)},
    ],

    # ── GRANT ROAD / MARINE LINES / CHURCHGATE ────────────────────────────────
    "grant road": [
        {"id":"GR-01","address":"Dalamal Towers, Flat 1002, Grant Road West, Mumbai-400007",
         "bedrooms":2,"bathrooms":2,"sqft":1200,"carpet_area_sqft":900,"type":"flat","status":"for sale",
         "floor":"10th of 18","society":"Dalamal Towers","possession":"Ready to Move","rera_id":"P51800001890",
         "price_per_sqft":"Rs. 38,000/sqft","days_on_market":29,"zone":"south",
         "description":"2 BHK in Dalamal Towers, Grant Road West — near Kemps Corner, Mahalaxmi Temple, Grant Road station.",
         "features":["city view","modular kitchen","covered parking","power backup"],
         "amenities":["gym","CCTV","24hr security","intercom"],"source":"demo",
         **_p(45_600_000)},
    ],
    "marine lines": [
        {"id":"ML2-01","address":"Marine Drive Apartments, Flat 802, Marine Drive, Marine Lines, Mumbai-400020",
         "bedrooms":2,"bathrooms":2,"sqft":1400,"carpet_area_sqft":1050,"type":"flat","status":"for sale",
         "floor":"8th of 14","society":"Marine Drive Apartments","possession":"Ready to Move","rera_id":"P51800002012",
         "price_per_sqft":"Rs. 55,000/sqft","days_on_market":45,"zone":"south",
         "description":"2 BHK on Marine Drive, Marine Lines — Queen's Necklace sea view, an iconic address in South Mumbai.",
         "features":["direct sea view","high ceilings","vintage charm","covered parking","cross ventilation"],
         "amenities":["watchman","generator","CCTV"],"source":"demo",
         **_p(77_000_000)},
    ],
    "churchgate": [
        {"id":"CG-01","address":"Jolly Maker Chambers II, Flat 1104, Nariman Point, Churchgate, Mumbai-400021",
         "bedrooms":3,"bathrooms":3,"sqft":2400,"carpet_area_sqft":1800,"type":"flat","status":"for sale",
         "floor":"11th of 20","society":"Jolly Maker Chambers","possession":"Ready to Move","rera_id":"P51800002234",
         "price_per_sqft":"Rs. 62,000/sqft","days_on_market":60,"zone":"south",
         "description":"3 BHK in Jolly Maker Chambers, Nariman Point area — South Mumbai's premier business district, sea-facing.",
         "features":["sea view","high ceilings","large living room","2 parking","heritage value"],
         "amenities":["watchman","generator","CCTV"],"source":"demo",
         **_p(148_800_000)},
    ],

    # ── GHATKOPAR ─────────────────────────────────────────────────────────────
    "ghatkopar": [
        {"id":"GK-01","address":"Dosti Acres, Tower F, Flat 1704, Ghatkopar East, Mumbai-400075",
         "bedrooms":2,"bathrooms":2,"sqft":1060,"carpet_area_sqft":795,"type":"flat","status":"for sale",
         "floor":"17th of 30","society":"Dosti Acres","possession":"Ready to Move","rera_id":"P51800021234",
         "price_per_sqft":"Rs. 20,500/sqft","days_on_market":18,"zone":"central",
         "description":"2 BHK in Dosti Acres, Ghatkopar East — at Metro Line 1 (Versova–Andheri–Ghatkopar) terminus, near R-City Mall.",
         "features":["metro proximity","modular kitchen","covered parking","EV charging","power backup"],
         "amenities":["swimming pool","gym","club","CCTV","children play area"],"source":"demo",
         **_p(21_730_000)},
        {"id":"GK-02","address":"Rustomjee Urbania Ghatkopar, Wing B, Flat 1204, Ghatkopar West, Mumbai-400086",
         "bedrooms":3,"bathrooms":2,"sqft":1380,"carpet_area_sqft":1035,"type":"flat","status":"for sale",
         "floor":"12th of 25","society":"Rustomjee Urbania Ghatkopar","possession":"Ready to Move","rera_id":"P51800021567",
         "price_per_sqft":"Rs. 22,000/sqft","days_on_market":31,"zone":"central",
         "description":"3 BHK in Rustomjee Urbania, Ghatkopar West — township model, near Metro & Central railway.",
         "features":["modular kitchen","2 parking","power backup","vastu compliant"],
         "amenities":["pool","gym","retail","school nearby","CCTV"],"source":"demo",
         **_p(30_360_000)},
        {"id":"GK-03","address":"Jain CHS, Flat 501, LBS Marg, Ghatkopar West, Mumbai-400086",
         "bedrooms":2,"bathrooms":2,"sqft":900,"carpet_area_sqft":675,"type":"flat","status":"for rent",
         "floor":"5th of 12","society":"Jain CHS","possession":"Immediate","rera_id":None,
         "price_per_sqft":"Rs. 40/sqft/month","days_on_market":4,"zone":"central",
         "description":"2 BHK on rent in Ghatkopar West — near Metro Line 1, R-City Mall, Central railway.",
         "features":["semi-furnished","fan/light fittings","modular kitchen","1 parking"],
         "amenities":["watchman","CCTV","garden"],"source":"demo",
         **_r(36_000)},
    ],

    # ── MULUND ────────────────────────────────────────────────────────────────
    "mulund": [
        {"id":"MU-01","address":"Lodha Aurum, Tower A, Flat 1204, Mulund West, Mumbai-400080",
         "bedrooms":2,"bathrooms":2,"sqft":1080,"carpet_area_sqft":810,"type":"flat","status":"for sale",
         "floor":"12th of 24","society":"Lodha Aurum","possession":"Ready to Move","rera_id":"P51800020789",
         "price_per_sqft":"Rs. 16,500/sqft","days_on_market":23,"zone":"central",
         "description":"2 BHK in Lodha Aurum, Mulund West — near Eastern Express Highway, 5 min from Mulund station.",
         "features":["modular kitchen","covered parking","power backup","EV charging"],
         "amenities":["swimming pool","gym","clubhouse","children play area","CCTV"],"source":"demo",
         **_p(17_820_000)},
        {"id":"MU-02","address":"Dosti Imperia, Wing B, Flat 804, Mulund East, Mumbai-400081",
         "bedrooms":3,"bathrooms":2,"sqft":1280,"carpet_area_sqft":960,"type":"flat","status":"for sale",
         "floor":"8th of 20","society":"Dosti Imperia","possession":"Ready to Move","rera_id":"P51800021012",
         "price_per_sqft":"Rs. 15,800/sqft","days_on_market":35,"zone":"central",
         "description":"3 BHK in Dosti Imperia, Mulund East — border of Thane, great connectivity on LBS Marg.",
         "features":["modular kitchen","2 parking","power backup","vastu compliant"],
         "amenities":["gym","garden","CCTV","children play area"],"source":"demo",
         **_p(20_224_000)},
    ],

    # ── VERSOVA ───────────────────────────────────────────────────────────────
    "versova": [
        {"id":"VS-01","address":"Sea Pearl, Flat 601, Yari Road, Versova, Andheri West, Mumbai-400061",
         "bedrooms":2,"bathrooms":2,"sqft":1150,"carpet_area_sqft":863,"type":"flat","status":"for sale",
         "floor":"6th of 12","society":"Sea Pearl CHS","possession":"Ready to Move","rera_id":"P51800006789",
         "price_per_sqft":"Rs. 28,000/sqft","days_on_market":26,"zone":"western",
         "description":"2 BHK in Sea Pearl, Versova — sea view, near Versova Metro station and Versova beach.",
         "features":["sea view","modular kitchen","covered parking","power backup"],
         "amenities":["gym","garden","CCTV","watchman"],"source":"demo",
         **_p(32_200_000)},
        {"id":"VS-02","address":"Mahindra Luminare, Flat 1104, JP Road, Versova, Mumbai-400061",
         "bedrooms":3,"bathrooms":3,"sqft":1680,"carpet_area_sqft":1260,"type":"flat","status":"for rent",
         "floor":"11th of 22","society":"Mahindra Luminare","possession":"Immediate","rera_id":None,
         "price_per_sqft":"Rs. 70/sqft/month","days_on_market":8,"zone":"western",
         "description":"3 BHK on rent in Mahindra Luminare, Versova — fully furnished, Versova Metro walkable.",
         "features":["fully furnished","AC in all rooms","modular kitchen","2 parking"],
         "amenities":["swimming pool","gym","CCTV","24hr security"],"source":"demo",
         **_r(117_000)},
    ],

    # ── LOKHANDWALA ───────────────────────────────────────────────────────────
    "lokhandwala": [
        {"id":"LK-01","address":"Shri Balaji Horizon, Flat 1204, Lokhandwala Complex, Andheri West, Mumbai-400053",
         "bedrooms":3,"bathrooms":3,"sqft":1600,"carpet_area_sqft":1200,"type":"flat","status":"for sale",
         "floor":"12th of 22","society":"Shri Balaji Horizon","possession":"Ready to Move","rera_id":"P51800007890",
         "price_per_sqft":"Rs. 30,000/sqft","days_on_market":28,"zone":"western",
         "description":"3 BHK in Lokhandwala Complex, Andheri West — heart of Lokhandwala, all amenities walkable.",
         "features":["modular kitchen","2 parking","power backup","vastu compliant"],
         "amenities":["gym","garden","CCTV","intercom"],"source":"demo",
         **_p(48_000_000)},
    ],

    # ── VILE PARLE ────────────────────────────────────────────────────────────
    "vile parle": [
        {"id":"VP-01","address":"Platinum Business Park, Flat 904, Andheri-Kurla Road, Vile Parle East, Mumbai-400099",
         "bedrooms":2,"bathrooms":2,"sqft":1100,"carpet_area_sqft":825,"type":"flat","status":"for sale",
         "floor":"9th of 18","society":"Platinum Heights","possession":"Ready to Move","rera_id":"P51800008234",
         "price_per_sqft":"Rs. 30,000/sqft","days_on_market":22,"zone":"western",
         "description":"2 BHK in Vile Parle East — airport proximity (7 min), near Metro Line 1, Western Railway.",
         "features":["modular kitchen","covered parking","power backup","vastu compliant"],
         "amenities":["gym","CCTV","garden","intercom"],"source":"demo",
         **_p(33_000_000)},
        {"id":"VP-02","address":"Sunder Mahal, Flat 302, SV Road, Vile Parle West, Mumbai-400056",
         "bedrooms":2,"bathrooms":2,"sqft":1050,"carpet_area_sqft":788,"type":"flat","status":"for rent",
         "floor":"3rd of 8","society":"Sunder Mahal","possession":"Immediate","rera_id":None,
         "price_per_sqft":"Rs. 68/sqft/month","days_on_market":5,"zone":"western",
         "description":"2 BHK on rent in Vile Parle West — near Vile Parle station, old building, great location.",
         "features":["semi-furnished","AC in living room","modular kitchen"],
         "amenities":["watchman","CCTV"],"source":"demo",
         **_r(71_000)},
    ],

    # ── SANTACRUZ ─────────────────────────────────────────────────────────────
    "santacruz": [
        {"id":"SC-01","address":"Suraj Vihar, Flat 802, Perry Road, Santacruz West, Mumbai-400054",
         "bedrooms":3,"bathrooms":3,"sqft":1680,"carpet_area_sqft":1260,"type":"flat","status":"for sale",
         "floor":"8th of 14","society":"Suraj Vihar","possession":"Ready to Move","rera_id":"P51800009123",
         "price_per_sqft":"Rs. 36,000/sqft","days_on_market":30,"zone":"western",
         "description":"3 BHK in Suraj Vihar, Santacruz West — quiet residential street, 5 min to Santacruz station.",
         "features":["modular kitchen","covered parking","power backup","vastu compliant"],
         "amenities":["gym","garden","watchman","CCTV"],"source":"demo",
         **_p(60_480_000)},
    ],

    # ── KHAR ──────────────────────────────────────────────────────────────────
    "khar": [
        {"id":"KH-01","address":"Amrita CHS, Flat 601, 14th Road, Khar West, Mumbai-400052",
         "bedrooms":3,"bathrooms":3,"sqft":1750,"carpet_area_sqft":1313,"type":"flat","status":"for sale",
         "floor":"6th of 10","society":"Amrita CHS","possession":"Ready to Move","rera_id":"P51800005678",
         "price_per_sqft":"Rs. 48,000/sqft","days_on_market":40,"zone":"western",
         "description":"3 BHK in Khar West on 14th Road — quiet bungalow lane, Khar Danda sea proximity.",
         "features":["modular kitchen","covered parking","power backup","cross ventilation"],
         "amenities":["watchman","CCTV","garden"],"source":"demo",
         **_p(84_000_000)},
    ],

    # ── MAHIM ─────────────────────────────────────────────────────────────────
    "mahim": [
        {"id":"MA-01","address":"Sugee Marina, Tower B, Flat 1104, Mahim West, Mumbai-400016",
         "bedrooms":2,"bathrooms":2,"sqft":1180,"carpet_area_sqft":885,"type":"flat","status":"for sale",
         "floor":"11th of 25","society":"Sugee Marina","possession":"Ready to Move","rera_id":"P51800005234",
         "price_per_sqft":"Rs. 38,000/sqft","days_on_market":26,"zone":"western",
         "description":"2 BHK in Sugee Marina, Mahim — partial sea view, Mahim Creek, close to Bandra-Mahim link.",
         "features":["partial sea view","modular kitchen","covered parking","power backup"],
         "amenities":["swimming pool","gym","CCTV","24hr security"],"source":"demo",
         **_p(44_840_000)},
    ],

    # ── PRABHADEVI ────────────────────────────────────────────────────────────
    "prabhadevi": [
        {"id":"PD-01","address":"Kalpataru Horizon, Wing A, Flat 2004, Prabhadevi, Mumbai-400025",
         "bedrooms":3,"bathrooms":3,"sqft":2100,"carpet_area_sqft":1575,"type":"flat","status":"for sale",
         "floor":"20th of 35","society":"Kalpataru Horizon","possession":"Ready to Move","rera_id":"P51800001567",
         "price_per_sqft":"Rs. 48,000/sqft","days_on_market":44,"zone":"south",
         "description":"3 BHK in Kalpataru Horizon, Prabhadevi — Siddhivinayak Temple proximity, sea-link views, BKC access.",
         "features":["sea link view","Italian marble","2 parking","smart home","servant quarters"],
         "amenities":["rooftop pool","spa","gym","concierge","valet parking"],"source":"demo",
         **_p(100_800_000)},
    ],

    # ── MAHALAXMI ─────────────────────────────────────────────────────────────
    "mahalaxmi": [
        {"id":"MH-01","address":"Saifee Burhani Upliftment, Flat 1202, Bhendi Bazaar, Mahalaxmi, Mumbai-400034",
         "bedrooms":2,"bathrooms":2,"sqft":1050,"carpet_area_sqft":788,"type":"flat","status":"for sale",
         "floor":"12th of 17","society":"SBUT Residential","possession":"Ready to Move","rera_id":"P51800001678",
         "price_per_sqft":"Rs. 40,000/sqft","days_on_market":36,"zone":"western",
         "description":"2 BHK near Mahalaxmi Temple — premium location, racecourse view, near Haji Ali.",
         "features":["racecourse view","modular kitchen","covered parking","power backup"],
         "amenities":["gym","CCTV","24hr security"],"source":"demo",
         **_p(42_000_000)},
    ],
}

# Aliases — map alternate spellings/station names to the canonical key
_ALIAS: dict = {
    "nalasopara": "nallasopara", "naigon": "naigaon",
    "bhayandar": "bhayander",
    "vasai road": "vasai", "vasai-virar": "virar",
    "virar west": "virar", "virar east": "virar",
    "khar road": "khar",
    "matunga road": "dadar",   # Matunga Road station is in Dadar belt
    "hiranandani": "powai",
    "lokhandwala complex": "lokhandwala",
    "versova andheri": "versova",
    "andheri west": "andheri", "andheri east": "andheri",
    "bandra west": "bandra", "bandra east": "bandra",
    "malad west": "malad",
    "goregaon west": "goregaon", "goregaon east": "goregaon",
    "kandivali west": "kandivali", "kandivali east": "kandivali",
    "borivali west": "borivali", "borivali east": "borivali",
    "ghatkopar west": "ghatkopar", "ghatkopar east": "ghatkopar",
    "chembur west": "chembur", "chembur east": "chembur",
    "mulund west": "mulund", "mulund east": "mulund",
    "dadar west": "dadar", "dadar east": "dadar",
    "wadala east": "wadala", "wadala road": "wadala",
    "lower parel west": "lower parel",
    "nariman point": "churchgate",
    "marine drive": "marine lines",
    "mira road east": "mira road", "mira bhayandar": "mira road",
    "dahisar east": "dahisar", "dahisar west": "dahisar",
}


def _mumbai_listings(location: str) -> list:
    """Return location-accurate listings from AREA_LISTINGS, or best-match fallback."""
    loc = location.lower().strip()

    # 1. Resolve alias
    for alias, canonical in _ALIAS.items():
        if alias in loc:
            loc = canonical
            break

    # 2. Exact key match
    if loc in AREA_LISTINGS:
        return AREA_LISTINGS[loc]

    # 3. Partial key match (e.g. "bandra west" → "bandra")
    for key in AREA_LISTINGS:
        if key in loc or loc in key:
            return AREA_LISTINGS[key]

    # 4. Zone-level fallback — pick representative area from same zone
    zone = _zone_for(location)
    zone_defaults = {
        "south":   "worli",
        "western": "andheri",
        "central": "powai",
        "harbour": "wadala",
    }
    return AREA_LISTINGS.get(zone_defaults[zone], AREA_LISTINGS["andheri"])


# ─── Tool Implementations ──────────────────────────────────────────────────────

def search_properties(location, min_price=None, max_price=None, bedrooms=None,
                      bathrooms=None, property_type="any", for_sale_or_rent="sale"):
    loc_clean = location.strip()
    if not is_mumbai_location(loc_clean):
        return {
            "error": "out_of_scope",
            "message": (
                f"'{loc_clean}' is outside our coverage area. "
                "This assistant covers only Mumbai — Western, Central, Harbour, and South Mumbai. "
                "Please ask about areas like Bandra, Andheri, Powai, Worli, Chembur, Dadar, Wadala, etc."
            )
        }

    for_rent = (for_sale_or_rent == "rent")

    # Try live API first; fall back to demo data
    live = _fetch_live_properties(
        loc_clean, bedrooms=bedrooms,
        min_price=min_price, max_price=max_price,
        for_rent=for_rent,
    )
    data_source = "live" if live else "demo"
    listings = live if live else _mumbai_listings(loc_clean)

    results = []
    for prop in listings:
        if for_sale_or_rent == "sale" and prop["status"] == "for rent":
            continue
        if for_sale_or_rent == "rent" and prop["status"] == "for sale":
            continue
        price = prop["price"]
        if min_price and price < min_price:
            continue
        if max_price and price > max_price:
            continue
        if bedrooms and prop["bedrooms"] < bedrooms:
            continue
        if bathrooms and isinstance(prop["bathrooms"], (int, float)) and prop["bathrooms"] < bathrooms:
            continue
        if property_type not in ("any", None):
            pt = property_type.lower()
            if pt in ("flat", "apartment") and prop["type"] in ("flat", "apartment"):
                pass
            elif prop["type"] != pt:
                continue
        results.append(prop)

    note = (
        "Live data from RapidAPI India property feed."
        if data_source == "live"
        else (
            "Demo data shown. For live listings set RAPIDAPI_KEY env var "
            "(free at rapidapi.com — search 'India Real Estate')."
        )
    )
    return {
        "location": loc_clean,
        "zone": _zone_for(loc_clean),
        "currency": "INR",
        "data_source": data_source,
        "total_results": len(results),
        "listings": results if results else listings[:3],
        "note": note,
    }


def calculate_emi(home_price, down_payment_percent, annual_interest_rate,
                  loan_term_years, property_tax_annual=None):
    down_payment = home_price * (down_payment_percent / 100)
    loan_amount = home_price - down_payment
    monthly_rate = (annual_interest_rate / 100) / 12
    num_payments = loan_term_years * 12

    if monthly_rate == 0:
        monthly_emi = loan_amount / num_payments
    else:
        monthly_emi = loan_amount * (
            monthly_rate * (1 + monthly_rate) ** num_payments
        ) / ((1 + monthly_rate) ** num_payments - 1)

    total_paid = monthly_emi * num_payments
    total_interest = total_paid - loan_amount
    monthly_tax = (property_tax_annual / 12) if property_tax_annual else 0
    total_monthly = monthly_emi + monthly_tax

    def fmt(v):
        if v >= 10_000_000:
            return f"Rs. {v/10_000_000:.2f} Cr"
        elif v >= 100_000:
            return f"Rs. {v/100_000:.2f} L"
        return f"Rs. {v:,.0f}"

    stamp_duty = home_price * 0.05
    registration = home_price * 0.01
    processing_fee_range = f"Rs. {home_price * 0.005/100000:.1f}L – {home_price * 0.01/100000:.1f}L"

    return {
        "currency": "INR",
        "home_price": fmt(home_price),
        "down_payment": fmt(round(down_payment, 2)),
        "down_payment_percent": f"{down_payment_percent}%",
        "loan_amount": fmt(round(loan_amount, 2)),
        "annual_interest_rate": f"{annual_interest_rate}%",
        "loan_tenure": f"{loan_term_years} years ({num_payments} EMIs)",
        "monthly_emi": fmt(round(monthly_emi, 2)),
        "monthly_property_tax": fmt(round(monthly_tax, 2)) if monthly_tax else "Not provided",
        "total_monthly_outflow": fmt(round(total_monthly, 2)),
        "total_interest_paid": fmt(round(total_interest, 2)),
        "total_loan_cost": fmt(round(total_paid, 2)),
        "one_time_costs": {
            "stamp_duty_5pct_maharashtra": fmt(stamp_duty),
            "registration_1pct": fmt(registration),
            "processing_fee_approx": processing_fee_range,
            "total_acquisition_cost_approx": fmt(home_price + stamp_duty + registration),
        },
        "tax_benefits": [
            "Section 24b: Deduction up to Rs. 2L/year on home loan interest",
            "Section 80C: Deduction up to Rs. 1.5L/year on principal repayment",
            "First-time buyer: Additional Rs. 50,000 deduction under Section 80EE (if applicable)",
        ],
        "current_rates": {
            "SBI": "8.50% p.a.", "HDFC": "8.70% p.a.",
            "ICICI": "8.75% p.a.", "Axis": "8.75% p.a.", "Kotak": "8.65% p.a."
        },
    }


MUMBAI_NEIGHBORHOOD_DATA = {
    # South Mumbai
    "colaba": {
        "overall_score": 95, "zone": "South Mumbai",
        "known_for": "Heritage precinct, Gateway of India, diplomatic enclave, upscale dining",
        "schools": ["Cathedral & John Connon (10/10)", "Bombay Scottish (9/10)", "BD Somani (9/10)"],
        "hospitals": ["Breach Candy Hospital (10 min)", "Bombay Hospital (12 min)"],
        "transit": {"local_train": "CSMT (20 min)", "metro": "Metro Line 3 (Cuffe Parade)", "score": "8/10"},
        "walkability": "Excellent — neighbourhood is compact and pedestrian-friendly",
        "real_estate": {
            "avg_price_2bhk": "Rs. 6–12 Cr", "avg_price_3bhk": "Rs. 10–25 Cr",
            "avg_rent_2bhk": "Rs. 1,50,000–3,00,000/month",
            "price_per_sqft": "Rs. 55,000–1,00,000", "price_trend_1yr": "+8.5%",
        },
        "pros": ["Ultra-premium address", "Heritage charm", "Quiet streets", "Great restaurants"],
        "cons": ["Very expensive", "Old buildings need renovation", "Limited new supply"],
    },
    "worli": {
        "overall_score": 93, "zone": "South-Central Mumbai",
        "known_for": "Sea Link landmark, luxury high-rises, BKC proximity, nightlife",
        "schools": ["Bombay International School (nearby)", "Arya Vidya Mandir (10 min)"],
        "hospitals": ["Breach Candy Hospital (15 min)", "Hinduja Hospital (12 min)"],
        "transit": {"local_train": "Lower Parel/Mahalaxmi (10 min)", "metro": "Metro Line 3 (upcoming)", "score": "8/10"},
        "walkability": "Good — seafront promenade, Worli Sea Face",
        "real_estate": {
            "avg_price_2bhk": "Rs. 5–10 Cr", "avg_price_3bhk": "Rs. 8–20 Cr",
            "avg_rent_2bhk": "Rs. 1,20,000–2,50,000/month",
            "price_per_sqft": "Rs. 45,000–80,000", "price_trend_1yr": "+9.3%",
        },
        "pros": ["Sea Link access", "Premium towers", "BKC proximity", "Luxury malls"],
        "cons": ["Very expensive", "Traffic on weekends", "Limited parking"],
    },
    "lower parel": {
        "overall_score": 90, "zone": "Central Mumbai",
        "known_for": "Corporate hub, premium malls (Palladium, High Street Phoenix), upscale dining",
        "schools": ["Cathedral & John Connon (nearby)", "Bombay Scottish (nearby)"],
        "hospitals": ["Wockhardt Hospital (5 min)", "KEM Hospital (10 min)"],
        "transit": {"local_train": "Lower Parel station (Western line)", "metro": "Metro Line 3 (Worli-Lower Parel)", "score": "9/10"},
        "walkability": "Very good — walkable to malls, restaurants, offices",
        "real_estate": {
            "avg_price_2bhk": "Rs. 4–8 Cr", "avg_price_3bhk": "Rs. 6–15 Cr",
            "avg_rent_2bhk": "Rs. 90,000–2,00,000/month",
            "price_per_sqft": "Rs. 35,000–65,000", "price_trend_1yr": "+8.8%",
        },
        "pros": ["Mill district revival", "Restaurant belt", "Office proximity", "Great nightlife"],
        "cons": ["Expensive", "Crowded on weekdays", "Parking shortage"],
    },
    "prabhadevi": {
        "overall_score": 88, "zone": "Central Mumbai",
        "known_for": "Siddhivinayak Temple, mid-point between South Mumbai & suburbs",
        "hospitals": ["Hinduja Hospital (10 min)", "KEM Hospital (15 min)"],
        "transit": {"local_train": "Prabhadevi station (Western line)", "metro": "Metro Line 3", "score": "9/10"},
        "walkability": "Very good",
        "real_estate": {
            "avg_price_2bhk": "Rs. 3.5–6 Cr", "avg_price_3bhk": "Rs. 5–10 Cr",
            "avg_rent_2bhk": "Rs. 75,000–1,50,000/month",
            "price_per_sqft": "Rs. 30,000–55,000", "price_trend_1yr": "+7.5%",
        },
        "pros": ["Religious landmark", "Central location", "Good connectivity"],
        "cons": ["Traffic around temple", "Limited green spaces"],
    },
    "bandra": {
        "overall_score": 94, "zone": "Western Line",
        "known_for": "The Queen of Suburbs — cafés, Bollywood celebrities, Bandstand, BKC proximity",
        "schools": ["St. Andrew's High School (9/10)", "Holy Family (8/10)", "St. Stanislaus (9/10)"],
        "hospitals": ["Lilavati Hospital (5 min)", "Holy Family Hospital (10 min)"],
        "transit": {"local_train": "Bandra (Western line)", "metro": "Metro Line 2A", "score": "9/10"},
        "walkability": "Excellent — Carter Road, Bandstand, Linking Road all walkable",
        "real_estate": {
            "avg_price_2bhk": "Rs. 3.5–6 Cr", "avg_price_3bhk": "Rs. 5–12 Cr",
            "avg_rent_2bhk": "Rs. 80,000–1,50,000/month",
            "price_per_sqft": "Rs. 35,000–55,000", "price_trend_1yr": "+7.2%",
        },
        "pros": ["Premium address", "Vibrant social scene", "BKC walking distance", "Great connectivity"],
        "cons": ["Very expensive", "Traffic congestion", "Parking scarcity"],
    },
    "juhu": {
        "overall_score": 91, "zone": "Western Line",
        "known_for": "Beachfront living, Bollywood stars, JVPD Scheme, fine dining",
        "schools": ["Ryan International (9/10)", "Juhu High School (7/10)"],
        "hospitals": ["Kokilaben Ambani Hospital (10 min)", "Nanavati Hospital (15 min)"],
        "transit": {"local_train": "Vile Parle (5 min)", "metro": "Metro Line 1", "score": "7/10"},
        "walkability": "Good — Juhu Beach walkable",
        "real_estate": {
            "avg_price_2bhk": "Rs. 4–7 Cr", "avg_price_3bhk": "Rs. 6–15 Cr",
            "avg_rent_2bhk": "Rs. 90,000–1,80,000/month",
            "price_per_sqft": "Rs. 40,000–65,000", "price_trend_1yr": "+6.5%",
        },
        "pros": ["Beachfront", "Quiet lanes", "Celebrity neighbourhood", "Low-rise character"],
        "cons": ["Limited metro connectivity", "Monsoon flooding risk", "Expensive"],
    },
    "andheri": {
        "overall_score": 85, "zone": "Western Line",
        "known_for": "Commercial hub, Andheri East MIDC, airport proximity, JVLR",
        "schools": ["St. Mary's School (8/10)", "DAV Public School (8/10)"],
        "hospitals": ["Seven Hills Hospital (10 min)", "Holy Spirit Hospital (15 min)"],
        "transit": {"local_train": "Andheri (Western + Harbour)", "metro": "Metro Line 1 & 2A & 7", "score": "9/10"},
        "walkability": "Good in West, moderate in East",
        "real_estate": {
            "avg_price_2bhk": "Rs. 1.5–2.5 Cr", "avg_price_3bhk": "Rs. 2–4 Cr",
            "avg_rent_2bhk": "Rs. 40,000–70,000/month",
            "price_per_sqft": "Rs. 18,000–28,000", "price_trend_1yr": "+6.8%",
        },
        "pros": ["Best metro connectivity in suburbs", "Airport proximity", "BKC accessible", "Affordable vs Bandra"],
        "cons": ["Crowded", "Traffic jams on JVLR", "Flooding in East"],
    },
    "powai": {
        "overall_score": 89, "zone": "Central Line",
        "known_for": "IT/tech hub, IIT Bombay, Hiranandani township, Powai Lake",
        "schools": ["Hiranandani Foundation School (9/10)", "Bombay Scottish (9/10)"],
        "hospitals": ["Hiranandani Hospital (5 min)", "Fortis Hospital (10 min)"],
        "transit": {"local_train": "Ghatkopar (Metro connection)", "metro": "Metro Line 6 (upcoming)", "score": "7/10"},
        "walkability": "Good — planned township with wide roads",
        "real_estate": {
            "avg_price_2bhk": "Rs. 1.8–3 Cr", "avg_price_3bhk": "Rs. 2.5–5 Cr",
            "avg_rent_2bhk": "Rs. 45,000–80,000/month",
            "price_per_sqft": "Rs. 20,000–32,000", "price_trend_1yr": "+8.1%",
        },
        "pros": ["IT hub", "Planned infrastructure", "Lake views", "IIT proximity", "Relatively affordable"],
        "cons": ["Far from South Mumbai", "Traffic to western suburbs", "Limited nightlife"],
    },
    "chembur": {
        "overall_score": 83, "zone": "Central/Harbour Line",
        "known_for": "Eastern Express Highway access, Monorail, BKC-adjacent, improving infrastructure",
        "schools": ["St. Anthony's High School (8/10)", "Atomic Energy Central School (8/10)"],
        "hospitals": ["Surana Sethia Hospital (10 min)", "Bombay Hospital (15 min)"],
        "transit": {"local_train": "Chembur (Harbour line)", "metro": "Metro Line 2B + Monorail", "score": "8/10"},
        "walkability": "Good — improving with new metro",
        "real_estate": {
            "avg_price_2bhk": "Rs. 1.5–2.5 Cr", "avg_price_3bhk": "Rs. 2–4 Cr",
            "avg_rent_2bhk": "Rs. 35,000–60,000/month",
            "price_per_sqft": "Rs. 18,000–28,000", "price_trend_1yr": "+9.5%",
        },
        "pros": ["Strong price appreciation", "BKC proximity", "Monorail + Metro", "Good schools"],
        "cons": ["Industrial legacy", "Traffic on EEH", "Less premium feel than west"],
    },
    "dadar": {
        "overall_score": 86, "zone": "Central Line",
        "known_for": "Central Mumbai hub, Shivaji Park, Marathi cultural heart, good connectivity",
        "schools": ["Balmohan Vidyamandir (9/10)", "IES School (8/10)"],
        "hospitals": ["Hinduja Hospital (5 min)", "KEM Hospital (10 min)"],
        "transit": {"local_train": "Dadar (Central + Western line interchange)", "metro": "Metro Line 3", "score": "10/10"},
        "walkability": "Excellent — Shivaji Park, market areas walkable",
        "real_estate": {
            "avg_price_2bhk": "Rs. 2.5–4 Cr", "avg_price_3bhk": "Rs. 3.5–7 Cr",
            "avg_rent_2bhk": "Rs. 55,000–1,00,000/month",
            "price_per_sqft": "Rs. 25,000–45,000", "price_trend_1yr": "+7.0%",
        },
        "pros": ["Best rail connectivity in Mumbai", "Central location", "Shivaji Park", "Cultural hub"],
        "cons": ["Crowded market areas", "Old building stock", "Parking difficulty"],
    },
    "wadala": {
        "overall_score": 80, "zone": "Harbour Line",
        "known_for": "Monorail terminus, BKC adjacent, rapidly developing, Eastern Freeway access",
        "schools": ["Atomic Energy Schools (nearby)", "St. Gregorios (10 min)"],
        "hospitals": ["KEM Hospital (15 min)", "Bhabha Hospital (10 min)"],
        "transit": {"local_train": "Wadala Road (Harbour line)", "metro": "Monorail + Metro Line 4 (upcoming)", "score": "8/10"},
        "walkability": "Moderate — developing area",
        "real_estate": {
            "avg_price_2bhk": "Rs. 1.5–2.5 Cr", "avg_price_3bhk": "Rs. 2–4 Cr",
            "avg_rent_2bhk": "Rs. 35,000–60,000/month",
            "price_per_sqft": "Rs. 18,000–30,000", "price_trend_1yr": "+10.2%",
        },
        "pros": ["BKC walking distance", "High appreciation potential", "Monorail", "Eastern Freeway"],
        "cons": ["Still developing", "Limited retail", "Industrial areas nearby"],
    },
    "malad": {
        "overall_score": 79, "zone": "Western Line",
        "known_for": "Malvani seafood, Mindspace IT park, Marve Beach, affordable suburbs",
        "schools": ["Ryan International (8/10)", "St. Xaviers (8/10)"],
        "hospitals": ["Criticare Hospital (10 min)", "Lifeline (15 min)"],
        "transit": {"local_train": "Malad (Western line)", "metro": "Metro Line 2A", "score": "8/10"},
        "walkability": "Good",
        "real_estate": {
            "avg_price_2bhk": "Rs. 1.2–2 Cr", "avg_price_3bhk": "Rs. 1.6–3 Cr",
            "avg_rent_2bhk": "Rs. 28,000–50,000/month",
            "price_per_sqft": "Rs. 15,000–22,000", "price_trend_1yr": "+6.2%",
        },
        "pros": ["Affordable", "IT park proximity", "Good metro connection", "Near beach"],
        "cons": ["Far from South Mumbai", "Traffic on SV Road"],
    },
    "ghatkopar": {
        "overall_score": 82, "zone": "Central Line",
        "known_for": "Metro interchange hub, Jain community, R-City Mall, strong connectivity",
        "hospitals": ["Rajawadi Hospital (5 min)", "Jaslok (30 min)"],
        "transit": {"local_train": "Ghatkopar (Central line)", "metro": "Metro Line 1 (Versova–Andheri–Ghatkopar)", "score": "10/10"},
        "walkability": "Very good",
        "real_estate": {
            "avg_price_2bhk": "Rs. 1.3–2.2 Cr", "avg_price_3bhk": "Rs. 2–3.5 Cr",
            "avg_rent_2bhk": "Rs. 30,000–55,000/month",
            "price_per_sqft": "Rs. 16,000–26,000", "price_trend_1yr": "+7.8%",
        },
        "pros": ["Best metro connectivity on Central line", "Commercial hub", "R-City Mall"],
        "cons": ["Crowded", "Older building stock in some areas"],
    },
}


def get_neighborhood_info(neighborhood: str):
    nb_key = neighborhood.lower().strip()

    # Try to match known data
    matched = None
    for key, data in MUMBAI_NEIGHBORHOOD_DATA.items():
        if key in nb_key or nb_key in key:
            matched = data
            break

    # Live geocoding + Walk Score
    live_scores = {}
    coords = _geocode(f"{neighborhood}, Mumbai")
    if coords:
        lat, lon = coords
        ws = _walk_score(f"{neighborhood}, Mumbai, India", lat, lon)
        if ws and ws.get("status") == 1:
            live_scores = {
                "walk_score": ws.get("walkscore"),
                "walk_desc": ws.get("description"),
                "transit_score": ws.get("transit", {}).get("score"),
                "transit_desc": ws.get("transit", {}).get("description"),
                "bike_score": ws.get("bike", {}).get("score"),
                "source": "Live — Walk Score API",
            }

    if not is_mumbai_location(nb_key):
        return {
            "error": "out_of_scope",
            "message": (
                f"'{neighborhood}' is outside our coverage. This assistant covers "
                "Mumbai's Western, Central, Harbour, and South Mumbai areas only."
            )
        }

    if matched:
        result = {"location": f"{neighborhood}, Mumbai", **matched}
    else:
        # Generic Mumbai response
        zone = _zone_for(nb_key)
        result = {
            "location": f"{neighborhood}, Mumbai",
            "zone": zone.title() + " Mumbai",
            "overall_score": 80,
            "known_for": f"Part of Mumbai's {zone.title()} line corridor",
            "transit": {
                "local_train": f"Mumbai Local — {zone.title()} line",
                "metro": "Expanding metro network",
            },
            "real_estate": {
                "avg_price_2bhk": "Rs. 1.2–3 Cr (varies by sub-area)",
                "avg_rent_2bhk": "Rs. 25,000–75,000/month",
                "price_trend_1yr": "+6–8%",
            },
        }

    if live_scores:
        result["live_walkability"] = live_scores

    return result


def get_market_trends(location: str, period: str = "1year"):
    trends_data = {
        "3months": {"price_change": "+2.8%", "new_launches": "Very High", "absorption": "74%"},
        "6months": {"price_change": "+5.2%", "new_launches": "Very High", "absorption": "71%"},
        "1year":   {"price_change": "+9.8%", "new_launches": "Record High", "absorption": "76%"},
        "3years":  {"price_change": "+34.5%", "new_launches": "Record High", "absorption": "79%"},
    }

    # Zone-specific overrides
    zone = _zone_for(location)
    zone_multipliers = {"south": 1.15, "western": 1.0, "central": 0.95, "harbour": 1.2}
    zm = zone_multipliers.get(zone, 1.0)

    t = trends_data.get(period, trends_data["1year"])
    base_pct = float(t["price_change"].replace("+", "").replace("%", ""))
    adjusted = f"+{base_pct * zm:.1f}%"

    if not is_mumbai_location(location):
        return {
            "error": "out_of_scope",
            "message": "Market trends are available only for Mumbai areas."
        }

    return {
        "location": f"{location} (Mumbai Metropolitan Region)",
        "zone": zone.title() + " Line",
        "currency": "INR",
        "period": period,
        "price_change": adjusted,
        "new_project_launches": t["new_launches"],
        "inventory_absorption_rate": t["absorption"],
        "avg_days_to_sell": "40–80 days",
        "market_type": "Seller's Market",
        "price_per_sqft_range": {
            "south": "Rs. 45,000–1,00,000 (Colaba/Worli/Lower Parel)",
            "western": "Rs. 18,000–55,000 (Andheri to Bandra)",
            "central": "Rs. 16,000–45,000 (Dadar to Ghatkopar/Powai)",
            "harbour": "Rs. 18,000–32,000 (Wadala/Chembur)",
        }.get(zone, "Rs. 15,000–80,000"),
        "forecast_next_12_months": "+8% to +13%",
        "top_performing_micro_markets": [
            "Wadala — highest appreciation on Harbour line",
            "Chembur — Metro Line 2B driving demand",
            "Worli — luxury supply constrained, prices rising",
            "Bandra West — perennial demand from HNIs & NRIs",
            "Powai — IT sector resurgence",
        ],
        "key_infrastructure_drivers": [
            "Metro Line 3 (Colaba–BKC–SEEPZ) — operational 2024",
            "Mumbai Coastal Road — reducing South Mumbai commute times",
            "Atal Setu (Mumbai Trans Harbour Link) — boosting Harbour line",
            "Metro Line 2A & 7 — transforming Western suburbs",
            "BKC Terminus (Bullet Train) — long-term catalyst",
        ],
        "home_loan_rates": {
            "SBI": "8.50% p.a.", "HDFC": "8.70% p.a.",
            "ICICI": "8.75% p.a.", "Axis": "8.75% p.a.", "Kotak": "8.65% p.a."
        },
        "note": "Data is simulated. Connect PropTiger / MagicBricks / NoBroker APIs for live data.",
    }


# ─── Tool Dispatcher ──────────────────────────────────────────────────────────

def execute_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "search_properties":
        result = search_properties(**tool_input)
    elif tool_name == "calculate_emi":
        # Coerce strings to numbers — LLMs sometimes pass numerics as strings
        emi_input = dict(tool_input)
        for key in ("home_price", "down_payment_percent", "annual_interest_rate",
                    "property_tax_annual", "insurance_annual"):
            if key in emi_input and emi_input[key] is not None:
                emi_input[key] = float(str(emi_input[key]).replace(",", "").replace("₹", "").strip())
        if "loan_term_years" in emi_input and emi_input["loan_term_years"] is not None:
            emi_input["loan_term_years"] = int(float(str(emi_input["loan_term_years"]).strip()))
        result = calculate_emi(**emi_input)
    elif tool_name == "get_neighborhood_info":
        result = get_neighborhood_info(**tool_input)
    elif tool_name == "get_market_trends":
        result = get_market_trends(**tool_input)
    else:
        result = {"error": f"Unknown tool: {tool_name}"}
    return json.dumps(result, indent=2)


# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Mumbai real estate advisor specialising exclusively in Mumbai's Western Line, Central Line, Harbour Line, and South Mumbai.

**Your coverage:**
- Western Line (Churchgate → Virar — all 28 stations): Churchgate, Marine Lines, Charni Road, Grant Road, Mumbai Central, Mahalaxmi, Lower Parel (Elphinstone), Prabhadevi, Dadar West, Matunga Road, Mahim, Bandra, Khar Road, Santacruz, Vile Parle, Andheri, Jogeshwari, Goregaon, Malad, Kandivali, Borivali, Dahisar, Mira Road, Bhayandar, Naigaon, Vasai Road, Nallasopara, Virar. Also: Juhu, Versova, Lokhandwala (off-station sub-areas)
- Central Line: Byculla, Parel, Dadar, Matunga, Sion, Kurla, Ghatkopar, Vikhroli, Kanjurmarg, Bhandup, Mulund, Chembur, Powai, Hiranandani
- Harbour Line: Wadala, Sewri, Mazgaon, Cotton Green, Govandi, Mankhurd
- South Mumbai: Colaba, Cuffe Parade, Nariman Point, Fort, Malabar Hill, Breach Candy, Pedder Road, Nepean Sea Road, Worli

**Mumbai expertise:**
- Property terminology: BHK, carpet area vs built-up area (RERA carpet), RERA registration, OC/CC certificate
- Indian home loans: SBI (~8.50%), HDFC (~8.70%), ICICI (~8.75%), Kotak (~8.65%)
- Maharashtra taxes: Stamp duty 5%, Registration 1%, Society charges (maintenance ~Rs. 3–8/sqft/month)
- Indian tax benefits: Section 24b (interest deduction up to Rs. 2L), Section 80C (principal up to Rs. 1.5L)
- Leading developers: Lodha, Godrej Properties, Oberoi Realty, Kalpataru, Rustomjee, Shapoorji Pallonji, Hiranandani, L&T Realty, Piramal Realty
- Property portals: 99acres, MagicBricks, Housing.com, NoBroker, Square Yards
- Mumbai infrastructure: Mumbai Metro lines, Mumbai Coastal Road, Atal Setu (MTHL), BKC

**Guidelines:**
- If a user asks about areas outside Mumbai (Delhi, Bangalore, US, etc.) — politely decline and redirect to Mumbai
- Always use INR — never USD or any other currency
- Proactively suggest which zone/neighbourhood fits the user's budget and lifestyle
- Always use tools — if a user mentions a location or budget, call search_properties or calculate_emi
- Mention RERA when recommending properties (always verify RERA registration)
- Explain carpet area vs built-up area when discussing flat sizes
- Mention stamp duty (5%) and registration (1%) for buyers
- Flag OC/CC certificate importance for under-construction projects
- Be warm, precise, and specific — quote exact price ranges not vague estimates
- For luxury queries, highlight Worli, Bandra, Juhu, Lower Parel, Colaba
- For budget queries, highlight Malad, Kandivali, Borivali, Mulund, Bhandup, Govandi
- For IT professionals: suggest Powai, Andheri East, Ghatkopar, BKC-adjacent areas
"""


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(title="Mumbai Real Estate AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)


class ChatRequest(BaseModel):
    messages: List[Any] = []
    user_message: str


def sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def stream_chat(messages: list):
    groq_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in messages:
        content = m["content"]
        if isinstance(content, list):
            text_parts = [b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "") for b in content]
            content = " ".join(p for p in text_parts if p)
        groq_messages.append({"role": m["role"], "content": content or ""})

    try:
        while True:
            full_text = ""
            tool_calls_map: dict = {}
            finish_reason = None

            stream = await client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=groq_messages,
                tools=GROQ_TOOLS,
                tool_choice="auto",
                max_tokens=4096,
                stream=True,
            )

            async for chunk in stream:
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                delta = choice.delta

                if delta.content:
                    full_text += delta.content
                    yield sse_event({"type": "text", "content": delta.content})

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        i = tc.index
                        if i not in tool_calls_map:
                            tool_calls_map[i] = {
                                "id": "", "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        if tc.id:
                            tool_calls_map[i]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_map[i]["function"]["name"] += tc.function.name
                            if tc.function.arguments:
                                tool_calls_map[i]["function"]["arguments"] += tc.function.arguments

            if finish_reason != "tool_calls":
                break

            tool_calls_list = [tool_calls_map[i] for i in sorted(tool_calls_map)]

            groq_messages.append({
                "role": "assistant",
                "content": full_text or None,
                "tool_calls": tool_calls_list,
            })

            for tc in tool_calls_list:
                name = tc["function"]["name"]
                try:
                    inp = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, ValueError):
                    inp = {}

                yield sse_event({"type": "tool_call", "name": name, "input": inp})
                result_str = execute_tool(name, inp)
                groq_messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str,
                })

        yield sse_event({"type": "done"})

    except Exception as exc:
        yield sse_event({"type": "error", "message": str(exc)})


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    messages = list(request.messages)
    messages.append({"role": "user", "content": request.user_message})
    return StreamingResponse(
        stream_chat(messages),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


app.mount("/static", StaticFiles(directory=static_dir), name="static")
