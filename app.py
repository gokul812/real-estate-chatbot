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


# ─── Property Listings ─────────────────────────────────────────────────────────

def _mumbai_listings(location: str) -> list:
    cfg = _area_cfg(location)
    area = cfg["label"]
    mult = cfg["mult"]
    zone = cfg["zone"]
    base = int(BASE_PRICE * mult)

    # Zone-specific builders
    if zone == "south":
        builders = ["Oberoi Realty", "Lodha", "Raheja Corp", "Kalpataru", "DB Realty"]
        features_pool = ["sea view", "panoramic view", "private terrace", "concierge",
                         "valet parking", "home automation", "Italian marble", "butler service"]
    elif zone == "western":
        builders = ["Godrej Properties", "Rustomjee", "Hiranandani", "Kalpataru", "L&T Realty"]
        features_pool = ["sea-facing", "modular kitchen", "club membership", "EV charging",
                         "rooftop terrace", "smart home", "covered parking", "2 parking"]
    elif zone == "central":
        builders = ["Lodha", "Shapoorji Pallonji", "Runwal", "Piramal", "Dosti Realty"]
        features_pool = ["lake view", "township amenities", "school nearby", "IT hub proximity",
                         "metro access", "modular kitchen", "gym", "covered parking"]
    else:  # harbour
        builders = ["Wadhwa Group", "Kanakia Group", "DB Realty", "Rajesh LifeSpaces", "Sugee Group"]
        features_pool = ["harbour view", "sea view", "metro access", "township living",
                         "modular kitchen", "gym", "covered parking", "24hr security"]

    def rera(): return f"P51800{__import__('random').randint(10000,99999)}"

    return [
        {
            "id": "MUM-001",
            "address": f"{builders[0]} Grandeur, Wing A, Flat 804, {area}, Mumbai",
            "price": int(base * 1.5),
            "price_display": _inr_fmt(int(base * 1.5)),
            "price_per_sqft": f"Rs. {int(base * 1.5 / 950):,}/sqft",
            "bedrooms": 2, "bathrooms": 2, "sqft": 950, "carpet_area_sqft": 710,
            "type": "flat", "status": "for sale", "days_on_market": 14,
            "floor": "8th of 24", "society": f"{builders[0]} Grandeur",
            "description": f"Elegant 2 BHK in {area} with premium finishes and {features_pool[0]}.",
            "features": features_pool[:4],
            "amenities": ["infinity pool", "clubhouse", "power backup", "CCTV", "intercom"],
            "possession": "Ready to Move", "rera_id": rera(), "zone": zone,
        },
        {
            "id": "MUM-002",
            "address": f"{builders[1]} Skyline, Tower B, Apt 1203, {area}, Mumbai",
            "price": int(base * 2.3),
            "price_display": _inr_fmt(int(base * 2.3)),
            "price_per_sqft": f"Rs. {int(base * 2.3 / 1450):,}/sqft",
            "bedrooms": 3, "bathrooms": 3, "sqft": 1450, "carpet_area_sqft": 1090,
            "type": "flat", "status": "for sale", "days_on_market": 22,
            "floor": "12th of 32", "society": f"{builders[1]} Skyline",
            "description": f"Spacious 3 BHK in {area} with city skyline views and luxury fittings.",
            "features": features_pool[1:5],
            "amenities": ["rooftop pool", "concierge", "EV charging", "indoor games", "spa"],
            "possession": "Ready to Move", "rera_id": rera(), "zone": zone,
        },
        {
            "id": "MUM-003",
            "address": f"{builders[2]} Signature, Wing C, {area}, Mumbai",
            "price": int(base * 4.2),
            "price_display": _inr_fmt(int(base * 4.2)),
            "price_per_sqft": f"Rs. {int(base * 4.2 / 2800):,}/sqft",
            "bedrooms": 4, "bathrooms": 4, "sqft": 2800, "carpet_area_sqft": 2100,
            "type": "penthouse" if zone == "south" else "flat",
            "status": "for sale", "days_on_market": 48,
            "floor": "22nd of 45", "society": f"{builders[2]} Signature",
            "description": f"Ultra-luxury 4 BHK residence in {area} with {features_pool[0]} and private terrace.",
            "features": features_pool[:5],
            "amenities": ["infinity pool", "helipad access", "valet parking", "spa", "private cinema"],
            "possession": "Ready to Move", "rera_id": rera(), "zone": zone,
        },
        {
            "id": "MUM-004",
            "address": f"{builders[3]} Heights, 3rd Floor, {area}, Mumbai",
            "price": int(base * 0.45),
            "price_display": _inr_fmt(int(base * 0.45)),
            "price_per_sqft": f"Rs. {int(base * 0.45 / 550):,}/sqft",
            "bedrooms": 1, "bathrooms": 1, "sqft": 550, "carpet_area_sqft": 415,
            "type": "studio" if base * 0.45 < 5_000_000 else "flat",
            "status": "for sale", "days_on_market": 6,
            "floor": "3rd of 18", "society": f"{builders[3]} Heights",
            "description": f"Compact 1 BHK in {area} — ideal for investment or first-time buyers.",
            "features": ["modular kitchen", "covered parking", "24hr water supply", "power backup"],
            "amenities": ["gym", "children's play area", "intercom"],
            "possession": "Ready to Move", "rera_id": rera(), "zone": zone,
        },
        {
            "id": "MUM-005",
            "address": f"{builders[4]} Residency, Wing A, {area}, Mumbai",
            "price": int(base * 0.38),
            "price_display": _inr_fmt(int(base * 0.38)) + "/month",
            "bedrooms": 2, "bathrooms": 2, "sqft": 1050, "carpet_area_sqft": 790,
            "type": "flat", "status": "for rent", "days_on_market": 4,
            "floor": "7th of 20", "society": f"{builders[4]} Residency",
            "description": f"Well-maintained 2 BHK on rent in {area}, semi-furnished with AC in all rooms.",
            "features": ["semi-furnished", "AC in all rooms", "modular kitchen", "1 parking"],
            "amenities": ["gym", "swimming pool", "24hr security", "CCTV"],
            "possession": "Immediate", "rera_id": None, "zone": zone,
        },
        {
            "id": "MUM-006",
            "address": f"Lodha Palava City, Apt 405, {area}, Mumbai",
            "price": int(base * 0.6),
            "price_display": _inr_fmt(int(base * 0.6)),
            "price_per_sqft": f"Rs. {int(base * 0.6 / 1100):,}/sqft",
            "bedrooms": 2, "bathrooms": 2, "sqft": 1100, "carpet_area_sqft": 825,
            "type": "flat", "status": "for sale", "days_on_market": 32,
            "floor": "4th of 22", "society": "Lodha Palava City",
            "description": f"2 BHK in integrated township in {area} — school, hospital & mall within complex.",
            "features": ["township living", "school nearby", "hospital nearby", "mall in complex"],
            "amenities": ["200+ amenities", "cricket ground", "multiple pools", "running track"],
            "possession": "Ready to Move", "rera_id": rera(), "zone": zone,
        },
    ]


def _inr_fmt(v: int) -> str:
    if v >= 10_000_000:
        cr = v / 10_000_000
        return f"Rs. {cr:.2f} Cr"
    elif v >= 100_000:
        l = v / 100_000
        return f"Rs. {l:.1f} L"
    else:
        return f"Rs. {v:,}"


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
