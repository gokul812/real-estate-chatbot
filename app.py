"""
Real Estate Chatbot — FastAPI web backend
Streams responses via Server-Sent Events with full tool-call loop.
Supports both US (USD) and India/Mumbai (INR) markets.
"""

import json
import asyncio
import os
import anthropic

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Any

# ─── Anthropic client ─────────────────────────────────────────────────────────

client = anthropic.Anthropic()

# ─── Tool Definitions ─────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "search_properties",
        "description": (
            "Search for properties based on location, price range, bedrooms, "
            "bathrooms, and property type. Supports India (Mumbai, Delhi, Bangalore, etc.) "
            "with INR pricing and US cities with USD pricing. Returns matching listings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City, neighborhood, or area (e.g. 'Bandra Mumbai', 'Powai', 'Austin TX')"
                },
                "min_price": {
                    "type": "number",
                    "description": "Minimum price. For India use INR (e.g. 5000000 for 50 lakhs). For US use USD."
                },
                "max_price": {
                    "type": "number",
                    "description": "Maximum price. For India use INR (e.g. 20000000 for 2 crore). For US use USD."
                },
                "bedrooms": {
                    "type": "integer",
                    "description": "Number of bedrooms/BHK (minimum)"
                },
                "bathrooms": {
                    "type": "number",
                    "description": "Number of bathrooms (minimum)"
                },
                "property_type": {
                    "type": "string",
                    "enum": ["flat", "apartment", "house", "villa", "penthouse", "studio", "plot", "any"],
                    "description": "Type of property. Use 'flat' or 'apartment' for India."
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
        "name": "calculate_mortgage",
        "description": (
            "Calculate monthly EMI/mortgage payments, total interest paid, and loan details. "
            "Works for both Indian home loans (INR, rates ~8.5-9.5%) and US mortgages (USD, rates ~6.5-7%). "
            "For India, 'mortgage' means Home Loan / EMI."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "home_price": {
                    "type": "number",
                    "description": "Total property price. INR for India (e.g. 10000000 for 1 crore), USD for US."
                },
                "down_payment_percent": {
                    "type": "number",
                    "description": "Down payment as a percentage (e.g. 20 for 20%). In India minimum is typically 10-20%."
                },
                "annual_interest_rate": {
                    "type": "number",
                    "description": "Annual interest rate as a percentage. India: ~8.5-9.5%. US: ~6.5-7.5%."
                },
                "loan_term_years": {
                    "type": "integer",
                    "description": "Loan term in years. India: up to 30 years. US: typically 15 or 30."
                },
                "currency": {
                    "type": "string",
                    "enum": ["INR", "USD"],
                    "description": "Currency for the calculation. Use INR for India, USD for US."
                },
                "property_tax_annual": {
                    "type": "number",
                    "description": "Annual property tax (optional)"
                },
                "insurance_annual": {
                    "type": "number",
                    "description": "Annual home insurance (optional)"
                }
            },
            "required": ["home_price", "down_payment_percent", "annual_interest_rate", "loan_term_years"]
        }
    },
    {
        "name": "get_neighborhood_info",
        "description": (
            "Get detailed information about a neighborhood including schools, "
            "crime rates, walkability, nearby amenities, and average home prices."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "neighborhood": {
                    "type": "string",
                    "description": "Neighborhood name or address"
                },
                "city": {
                    "type": "string",
                    "description": "City name"
                },
                "state": {
                    "type": "string",
                    "description": "State abbreviation (e.g. CA, TX, NY)"
                }
            },
            "required": ["neighborhood", "city"]
        }
    },
    {
        "name": "get_market_trends",
        "description": (
            "Get real estate market trends for a specific area including median "
            "home prices, days on market, price changes, and inventory levels."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City, county, or metro area"
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

# ─── Mock Tool Implementations ────────────────────────────────────────────────

MUMBAI_KEYWORDS = ["mumbai", "bombay", "bandra", "andheri", "juhu", "powai",
                   "worli", "lower parel", "malad", "kandivali", "borivali",
                   "thane", "navi mumbai", "kurla", "dadar", "goregaon",
                   "versova", "santacruz", "vile parle", "chembur", "mulund"]

INDIA_KEYWORDS = ["india", "delhi", "bangalore", "bengaluru", "hyderabad",
                  "chennai", "pune", "kolkata", "ahmedabad", "noida", "gurugram", "gurgaon"]


def is_india_location(location: str) -> bool:
    loc = location.lower()
    return any(k in loc for k in MUMBAI_KEYWORDS + INDIA_KEYWORDS)


def is_mumbai_location(location: str) -> bool:
    loc = location.lower()
    return any(k in loc for k in MUMBAI_KEYWORDS)


def search_properties(location, min_price=None, max_price=None, bedrooms=None,
                      bathrooms=None, property_type="any", for_sale_or_rent="sale"):
    """Property search with Mumbai/India and US listings."""

    if is_mumbai_location(location):
        listings = _mumbai_listings(location)
        currency = "INR"
        note = "Prices in INR. Data is simulated — connect 99acres/MagicBricks API for live listings."
    elif is_india_location(location):
        listings = _india_listings(location)
        currency = "INR"
        note = "Prices in INR. Data is simulated — connect 99acres/MagicBricks API for live listings."
    else:
        listings = _us_listings(location)
        currency = "USD"
        note = "Data is simulated — connect a real MLS/Zillow API for live listings."

    # Apply filters
    results = []
    for prop in listings:
        if for_sale_or_rent == "sale" and prop["status"] == "for rent":
            continue
        if for_sale_or_rent == "rent" and prop["status"] == "for sale":
            continue
        if min_price and prop["price"] < min_price:
            continue
        if max_price and prop["price"] > max_price:
            continue
        if bedrooms and prop["bedrooms"] < bedrooms:
            continue
        if bathrooms and prop["bathrooms"] < bathrooms:
            continue
        if property_type not in ("any", None) and prop["type"] not in (property_type, "any"):
            # loose match: flat == apartment
            if not (property_type in ("flat", "apartment") and prop["type"] in ("flat", "apartment")):
                continue
        results.append(prop)

    return {
        "location": location,
        "currency": currency,
        "total_results": len(results),
        "listings": results if results else listings[:3],
        "note": note
    }


def _mumbai_listings(location: str) -> list:
    """Realistic Mumbai property listings in INR."""
    loc = location.lower()

    # Determine sub-area for address realism
    if "bandra" in loc:
        area = "Bandra West"
        multiplier = 1.4
    elif "juhu" in loc:
        area = "Juhu"
        multiplier = 1.5
    elif "worli" in loc or "lower parel" in loc:
        area = "Worli"
        multiplier = 1.6
    elif "powai" in loc:
        area = "Powai"
        multiplier = 0.9
    elif "andheri" in loc:
        area = "Andheri West"
        multiplier = 0.85
    elif "thane" in loc:
        area = "Thane West"
        multiplier = 0.5
    elif "navi mumbai" in loc:
        area = "Navi Mumbai"
        multiplier = 0.45
    elif "malad" in loc or "kandivali" in loc or "borivali" in loc:
        area = "Kandivali West"
        multiplier = 0.7
    else:
        area = "Mumbai Suburbs"
        multiplier = 0.8

    base = int(10_000_000 * multiplier)   # 1 Cr base scaled by area

    return [
        {
            "id": "MUM-001",
            "address": f"Shree Sai CHS, Flat 804, Link Road, {area}, Mumbai",
            "price": int(base * 1.5),
            "price_display": f"Rs. {int(base * 1.5 / 100000)} Lakhs",
            "bedrooms": 2,
            "bathrooms": 2,
            "sqft": 950,
            "carpet_area_sqft": 720,
            "type": "flat",
            "status": "for sale",
            "days_on_market": 14,
            "floor": "8th of 22",
            "society": "Shree Sai CHS",
            "description": f"Spacious 2 BHK flat in {area} with sea-facing view, modular kitchen, and covered parking.",
            "features": ["sea view", "modular kitchen", "covered parking", "24hr security", "gym", "garden"],
            "amenities": ["swimming pool", "clubhouse", "power backup", "CCTV"],
            "possession": "Ready to Move",
            "rera_id": "P51800034123"
        },
        {
            "id": "MUM-002",
            "address": f"Raheja Residency, Tower B, Apt 1203, {area}, Mumbai",
            "price": int(base * 2.2),
            "price_display": f"Rs. {int(base * 2.2 / 100000)} Lakhs",
            "bedrooms": 3,
            "bathrooms": 3,
            "sqft": 1450,
            "carpet_area_sqft": 1100,
            "type": "flat",
            "status": "for sale",
            "days_on_market": 21,
            "floor": "12th of 30",
            "society": "Raheja Residency",
            "description": f"Premium 3 BHK in {area} with city skyline views, Italian marble flooring, and 2 parking slots.",
            "features": ["city view", "Italian marble", "2 parking slots", "servant quarters", "vastu compliant"],
            "amenities": ["rooftop pool", "indoor games", "concierge", "EV charging"],
            "possession": "Ready to Move",
            "rera_id": "P51800045678"
        },
        {
            "id": "MUM-003",
            "address": f"Oberoi Exquisite, Wing C, {area}, Mumbai",
            "price": int(base * 4.0),
            "price_display": f"Rs. {int(base * 4.0 / 100000)} Lakhs",
            "bedrooms": 4,
            "bathrooms": 4,
            "sqft": 2800,
            "carpet_area_sqft": 2100,
            "type": "flat",
            "status": "for sale",
            "days_on_market": 45,
            "floor": "20th of 40",
            "society": "Oberoi Exquisite",
            "description": f"Ultra-luxury 4 BHK residence in {area} with panoramic views, private terrace, and butler service.",
            "features": ["panoramic view", "private terrace", "butler service", "home automation", "wine cellar"],
            "amenities": ["infinity pool", "spa", "helipad access", "valet parking"],
            "possession": "Ready to Move",
            "rera_id": "P51800056789"
        },
        {
            "id": "MUM-004",
            "address": f"Arihant Aura, 3rd Floor, {area}, Mumbai",
            "price": int(base * 0.4),
            "price_display": f"Rs. {int(base * 0.4 / 100000)} Lakhs",
            "bedrooms": 1,
            "bathrooms": 1,
            "sqft": 550,
            "carpet_area_sqft": 415,
            "type": "flat",
            "status": "for sale",
            "days_on_market": 7,
            "floor": "3rd of 15",
            "society": "Arihant Aura",
            "description": f"Compact 1 BHK starter home in {area}, great for investment or first-time buyers.",
            "features": ["modular kitchen", "covered parking", "24hr water supply"],
            "amenities": ["gym", "children's play area", "intercom"],
            "possession": "Ready to Move",
            "rera_id": "P51800067890"
        },
        {
            "id": "MUM-005",
            "address": f"Godrej Prime, Wing A, {area}, Mumbai",
            "price": int(base * 0.35),   # monthly rent
            "price_display": f"Rs. {int(base * 0.35 / 100000)} Lakhs/month",
            "bedrooms": 2,
            "bathrooms": 2,
            "sqft": 1050,
            "carpet_area_sqft": 800,
            "type": "flat",
            "status": "for rent",
            "days_on_market": 5,
            "floor": "7th of 18",
            "society": "Godrej Prime",
            "description": f"Well-maintained 2 BHK flat on rent in {area}, semi-furnished with AC in all rooms.",
            "features": ["semi-furnished", "AC in all rooms", "modular kitchen", "1 parking"],
            "amenities": ["gym", "swimming pool", "24hr security"],
            "possession": "Immediate",
            "rera_id": None
        },
        {
            "id": "MUM-006",
            "address": f"Lodha Palava, Apartment 405, {area}, Mumbai",
            "price": int(base * 0.55),
            "price_display": f"Rs. {int(base * 0.55 / 100000)} Lakhs",
            "bedrooms": 2,
            "bathrooms": 2,
            "sqft": 1100,
            "carpet_area_sqft": 830,
            "type": "flat",
            "status": "for sale",
            "days_on_market": 30,
            "floor": "4th of 20",
            "society": "Lodha Palava",
            "description": f"Affordable 2 BHK by Lodha in {area} with township amenities — school, hospital & mall within complex.",
            "features": ["township living", "school nearby", "hospital nearby", "mall in complex"],
            "amenities": ["200+ amenities", "cricket ground", "multiple pools"],
            "possession": "Under Construction — Dec 2025",
            "rera_id": "P51800078901"
        }
    ]


def _india_listings(location: str) -> list:
    """Generic India listings for non-Mumbai cities."""
    return [
        {
            "id": "IND-001",
            "address": f"Prestige Towers, Flat 602, {location}",
            "price": 8_500_000,
            "price_display": "Rs. 85 Lakhs",
            "bedrooms": 2,
            "bathrooms": 2,
            "sqft": 1100,
            "type": "flat",
            "status": "for sale",
            "days_on_market": 18,
            "description": f"2 BHK flat in a premium gated society in {location} with all modern amenities.",
            "features": ["modular kitchen", "covered parking", "power backup", "24hr security"]
        },
        {
            "id": "IND-002",
            "address": f"Brigade Meadows, Villa 12, {location}",
            "price": 18_000_000,
            "price_display": "Rs. 1.8 Crore",
            "bedrooms": 4,
            "bathrooms": 4,
            "sqft": 2800,
            "type": "villa",
            "status": "for sale",
            "days_on_market": 35,
            "description": f"Spacious independent villa in {location} with private garden, 3-car garage.",
            "features": ["private garden", "3-car garage", "home theater", "terrace garden"]
        },
        {
            "id": "IND-003",
            "address": f"Sobha City, Apt 204, {location}",
            "price": 32_000,
            "price_display": "Rs. 32,000/month",
            "bedrooms": 2,
            "bathrooms": 2,
            "sqft": 950,
            "type": "flat",
            "status": "for rent",
            "days_on_market": 4,
            "description": f"2 BHK on rent in {location}, semi-furnished, walking distance to metro.",
            "features": ["semi-furnished", "metro access", "covered parking", "24hr security"]
        }
    ]


def _us_listings(location: str) -> list:
    """US property listings in USD."""
    return [
        {
            "id": "US-001",
            "address": f"142 Oak Street, {location}",
            "price": 485000,
            "price_display": "$485,000",
            "bedrooms": 3,
            "bathrooms": 2,
            "sqft": 1850,
            "type": "house",
            "status": "for sale",
            "days_on_market": 12,
            "description": "Charming 3BR/2BA ranch with updated kitchen, large backyard, 2-car garage.",
            "features": ["updated kitchen", "hardwood floors", "2-car garage", "large backyard"]
        },
        {
            "id": "US-002",
            "address": f"88 Maple Ave Unit 4B, {location}",
            "price": 320000,
            "price_display": "$320,000",
            "bedrooms": 2,
            "bathrooms": 2,
            "sqft": 1100,
            "type": "apartment",
            "status": "for sale",
            "days_on_market": 5,
            "description": "Modern condo in the heart of downtown with city views and rooftop access.",
            "features": ["city views", "rooftop deck", "in-unit laundry", "gym"]
        },
        {
            "id": "US-003",
            "address": f"310 Riverside Drive, {location}",
            "price": 725000,
            "price_display": "$725,000",
            "bedrooms": 4,
            "bathrooms": 3,
            "sqft": 2800,
            "type": "house",
            "status": "for sale",
            "days_on_market": 28,
            "description": "Spacious 4BR/3BA colonial with finished basement and mature trees.",
            "features": ["finished basement", "master suite", "fireplace", "deck"]
        },
        {
            "id": "US-004",
            "address": f"55 Pine Street, {location}",
            "price": 2200,
            "price_display": "$2,200/month",
            "bedrooms": 2,
            "bathrooms": 1,
            "sqft": 950,
            "type": "apartment",
            "status": "for rent",
            "days_on_market": 3,
            "description": "Bright 2BR apartment near transit, pet-friendly, utilities included.",
            "features": ["utilities included", "pet-friendly", "near transit", "storage"]
        }
    ]


def calculate_mortgage(home_price, down_payment_percent, annual_interest_rate,
                        loan_term_years, currency="USD",
                        property_tax_annual=None, insurance_annual=None):
    """Accurate EMI/mortgage calculator for both INR (India) and USD (US)."""
    currency = (currency or "USD").upper()
    is_inr = (currency == "INR")

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
    monthly_insurance = (insurance_annual / 12) if insurance_annual else 0
    total_monthly = monthly_emi + monthly_tax + monthly_insurance

    # India: no PMI concept but LTV > 80% attracts higher scrutiny
    pmi = 0
    if not is_inr and down_payment_percent < 20:
        pmi = loan_amount * 0.005 / 12
        total_monthly += pmi

    def fmt(v):
        if is_inr:
            if v >= 10_000_000:
                return f"Rs. {v/10_000_000:.2f} Cr (Rs. {v:,.0f})"
            elif v >= 100_000:
                return f"Rs. {v/100_000:.2f} L (Rs. {v:,.0f})"
            else:
                return f"Rs. {v:,.0f}"
        return f"${v:,.2f}"

    result = {
        "currency": currency,
        "home_price": fmt(home_price),
        "down_payment": fmt(round(down_payment, 2)),
        "down_payment_percent": f"{down_payment_percent}%",
        "loan_amount": fmt(round(loan_amount, 2)),
        "annual_interest_rate": f"{annual_interest_rate}%",
        "loan_term": f"{loan_term_years} years ({num_payments} EMIs)" if is_inr else f"{loan_term_years} years",
        "monthly_emi" if is_inr else "monthly_principal_interest": fmt(round(monthly_emi, 2)),
        "monthly_property_tax": fmt(round(monthly_tax, 2)) if monthly_tax else "Not provided",
        "monthly_insurance": fmt(round(monthly_insurance, 2)) if monthly_insurance else "Not provided",
        "total_monthly_payment": fmt(round(total_monthly, 2)),
        "total_interest_paid": fmt(round(total_interest, 2)),
        "total_cost_of_loan": fmt(round(total_paid, 2)),
    }

    if pmi:
        result["monthly_pmi"] = fmt(round(pmi, 2))
        result["pmi_note"] = "PMI applies because down payment is under 20%."

    if is_inr:
        result["india_notes"] = [
            "Tax benefit: Up to Rs. 2L/year deduction on interest (Section 24b)",
            "Tax benefit: Up to Rs. 1.5L/year deduction on principal (Section 80C)",
            "Home loan rates: SBI ~8.50%, HDFC ~8.70%, ICICI ~8.75% (check current rates)",
            "Processing fee: typically 0.5-1% of loan amount",
            "Stamp duty: 5-6% of property value (varies by state)"
        ]

    return result


MUMBAI_NEIGHBORHOOD_DATA = {
    "bandra": {
        "overall_score": 92,
        "known_for": "The Queen of Suburbs — upscale cafes, nightlife, Bollywood celebrities",
        "schools": ["St. Andrew's High School (9/10)", "Holy Family High School (8/10)", "Bandra Kurla Complex schools"],
        "hospitals": ["Lilavati Hospital (5 min)", "Holy Family Hospital (10 min)"],
        "transit": {"local_train": "Bandra station (Western line)", "metro": "Metro Line 2A nearby", "score": "9/10"},
        "walkability": "Excellent — most errands on foot",
        "real_estate": {
            "avg_price_2bhk": "Rs. 3.5–6 Cr",
            "avg_price_3bhk": "Rs. 5–12 Cr",
            "avg_rent_2bhk": "Rs. 80,000–1,50,000/month",
            "price_per_sqft": "Rs. 35,000–55,000",
            "price_trend_1yr": "+7.2%"
        },
        "pros": ["Premium location", "Great connectivity", "Vibrant social life", "Good schools"],
        "cons": ["Very expensive", "Traffic congestion", "Limited parking", "High cost of living"]
    },
    "juhu": {
        "overall_score": 90,
        "known_for": "Beachfront locality, Bollywood stars, upscale restaurants",
        "schools": ["Ryan International (9/10)", "Juhu High School (7/10)"],
        "hospitals": ["Kokilaben Dhirubhai Ambani Hospital (10 min)", "Nanavati Hospital (15 min)"],
        "transit": {"local_train": "Vile Parle station (5 min)", "metro": "Metro Line 1 nearby", "score": "7/10"},
        "walkability": "Good — Juhu Beach walkable",
        "real_estate": {
            "avg_price_2bhk": "Rs. 4–7 Cr",
            "avg_price_3bhk": "Rs. 6–15 Cr",
            "avg_rent_2bhk": "Rs. 90,000–1,80,000/month",
            "price_per_sqft": "Rs. 40,000–65,000",
            "price_trend_1yr": "+6.5%"
        },
        "pros": ["Beachfront", "Celebrity neighborhood", "Great restaurants", "Quiet streets"],
        "cons": ["Limited metro connectivity", "Expensive", "Flooding risk in monsoon"]
    },
    "powai": {
        "overall_score": 88,
        "known_for": "IT/tech hub, Hiranandani township, IIT Mumbai, lakeside living",
        "schools": ["Hiranandani Foundation School (9/10)", "Bombay Scottish (8/10)", "IIT Mumbai (research)"],
        "hospitals": ["Hiranandani Hospital (5 min)", "Fortis Hospital (10 min)"],
        "transit": {"local_train": "Ghatkopar/Kanjurmarg (10 min)", "metro": "Metro Line 6 (upcoming)", "score": "7/10"},
        "walkability": "Good — planned township with wide roads",
        "real_estate": {
            "avg_price_2bhk": "Rs. 1.8–3 Cr",
            "avg_price_3bhk": "Rs. 2.5–5 Cr",
            "avg_rent_2bhk": "Rs. 45,000–80,000/month",
            "price_per_sqft": "Rs. 20,000–32,000",
            "price_trend_1yr": "+8.1%"
        },
        "pros": ["IT hub", "Planned infrastructure", "Lake views", "Good schools", "Relatively affordable"],
        "cons": ["Far from South Mumbai", "Traffic to western suburbs", "Limited nightlife"]
    },
    "andheri": {
        "overall_score": 84,
        "known_for": "Commercial hub, airport proximity, metro connectivity",
        "schools": ["St. Mary's School (8/10)", "DAV Public School (8/10)"],
        "hospitals": ["Seven Hills Hospital (10 min)", "Holy Spirit Hospital (15 min)"],
        "transit": {"local_train": "Andheri station (Western + Harbour line)", "metro": "Metro Line 1 & 2A", "score": "9/10"},
        "walkability": "Good in Andheri West, moderate in East",
        "real_estate": {
            "avg_price_2bhk": "Rs. 1.5–2.5 Cr",
            "avg_price_3bhk": "Rs. 2–4 Cr",
            "avg_rent_2bhk": "Rs. 40,000–70,000/month",
            "price_per_sqft": "Rs. 18,000–28,000",
            "price_trend_1yr": "+6.8%"
        },
        "pros": ["Excellent connectivity", "Airport proximity", "Affordable vs Bandra", "BKC accessible"],
        "cons": ["Crowded", "Traffic jams", "Flooding in Andheri East"]
    },
    "worli": {
        "overall_score": 91,
        "known_for": "South Mumbai premium, sea-link access, luxury high-rises",
        "schools": ["Worli Municipal School", "Bombay International School (nearby)"],
        "hospitals": ["Breach Candy Hospital (20 min)", "Hinduja Hospital (15 min)"],
        "transit": {"local_train": "Mahalaxmi/Lower Parel (10 min)", "metro": "Metro Line 3 (upcoming)", "score": "8/10"},
        "walkability": "Good — seafront promenade",
        "real_estate": {
            "avg_price_2bhk": "Rs. 5–10 Cr",
            "avg_price_3bhk": "Rs. 8–20 Cr",
            "avg_rent_2bhk": "Rs. 1,20,000–2,50,000/month",
            "price_per_sqft": "Rs. 45,000–80,000",
            "price_trend_1yr": "+9.3%"
        },
        "pros": ["Sea-link access", "Premium location", "High-end malls", "BKC proximity"],
        "cons": ["Very expensive", "Limited green spaces", "Traffic"]
    }
}


def get_neighborhood_info(neighborhood, city, state=None):
    """Neighborhood data with Mumbai-specific intelligence."""
    location_str = f"{neighborhood}, {city}" + (f", {state}" if state else "")
    combined = f"{neighborhood} {city}".lower()

    # Check for Mumbai-specific neighborhood data
    if is_mumbai_location(combined) or is_mumbai_location(neighborhood):
        for key, data in MUMBAI_NEIGHBORHOOD_DATA.items():
            if key in neighborhood.lower():
                return {
                    "location": location_str,
                    "city": "Mumbai, Maharashtra, India",
                    **data,
                    "note": "Data is simulated. For live data connect to MagicBricks/99acres APIs."
                }
        # Generic Mumbai response
        return {
            "location": location_str,
            "city": "Mumbai, Maharashtra, India",
            "overall_score": 85,
            "known_for": "Part of Mumbai's diverse real estate market",
            "transit": {"local_train": "Mumbai Local — extensive network", "metro": "Expanding metro network", "score": "8/10"},
            "real_estate": {
                "avg_price_2bhk": "Rs. 1.5–5 Cr (varies by sub-area)",
                "avg_rent_2bhk": "Rs. 35,000–1,00,000/month",
                "price_trend_1yr": "+6–8%"
            },
            "note": "Data is simulated. For specific neighborhood deep-dives, ask about Bandra, Juhu, Powai, Andheri, or Worli."
        }

    # Generic response for other cities
    return {
        "location": location_str,
        "overall_score": 82,
        "schools": {
            "rating": "8/10",
            "nearby": [
                {"name": f"{neighborhood} School", "rating": "9/10", "distance": "0.3 mi"},
                {"name": f"{city} High School", "rating": "7/10", "distance": "1.2 mi"}
            ]
        },
        "safety": {"crime_rate": "below average", "safety_score": "7.8/10"},
        "walkability": {"walk_score": 74, "transit_score": 55, "rating": "Very Walkable"},
        "nearby_amenities": {"grocery_stores": 4, "restaurants": 23, "parks": 3, "hospitals": 1},
        "real_estate": {
            "median_home_price": "$465,000",
            "median_rent_2br": "$1,950/mo",
            "price_trend_1yr": "+4.2%",
            "avg_days_on_market": 18
        },
        "note": "Data is simulated for demonstration purposes."
    }


def get_market_trends(location, period="1year"):
    """Market trends for both Mumbai/India and US markets."""
    if is_mumbai_location(location):
        trends = {
            "3months": {"price_change": "+2.4%", "new_launches": "High", "absorption": "72%"},
            "6months": {"price_change": "+4.8%", "new_launches": "High", "absorption": "68%"},
            "1year":   {"price_change": "+9.1%", "new_launches": "Very High", "absorption": "74%"},
            "3years":  {"price_change": "+31.5%", "new_launches": "Very High", "absorption": "78%"},
        }
        t = trends.get(period, trends["1year"])
        return {
            "location": f"{location} (Mumbai Metropolitan Region)",
            "currency": "INR",
            "period": period,
            "median_flat_price_2bhk": "Rs. 1.8 Cr (suburbs) to Rs. 8 Cr (prime areas)",
            "price_change": t["price_change"],
            "new_project_launches": t["new_launches"],
            "inventory_absorption_rate": t["absorption"],
            "avg_days_to_sell": "45–90 days",
            "market_type": "Seller's Market",
            "price_per_sqft_range": "Rs. 18,000 (Thane) to Rs. 80,000 (Worli/Bandra)",
            "forecast_next_12mo": "+7% to +12%",
            "top_performing_areas": ["Bandra West", "Worli", "Powai", "Thane", "Navi Mumbai"],
            "hottest_segments": ["2 BHK under Rs. 1.5 Cr", "Luxury 3 BHK in BKC/Worli"],
            "key_drivers": [
                "Infra: Metro Line 3 (Colaba-SEEPZ) boosting nearby property values",
                "Infra: Mumbai Coastal Road reducing commute times",
                "Policy: RERA compliance increasing buyer confidence",
                "Demand: NRI investment surge post-pandemic",
                "Supply: Major launches by Lodha, Godrej, Oberoi, Prestige"
            ],
            "home_loan_rates": {
                "SBI": "8.50% p.a.",
                "HDFC": "8.70% p.a.",
                "ICICI": "8.75% p.a.",
                "Axis": "8.75% p.a."
            },
            "note": "Data is simulated. Connect PropTiger/MagicBricks/NoBroker APIs for live data."
        }
    elif is_india_location(location):
        trends = {
            "3months": {"price_change": "+2.1%"},
            "6months": {"price_change": "+4.2%"},
            "1year":   {"price_change": "+7.8%"},
            "3years":  {"price_change": "+24.3%"},
        }
        t = trends.get(period, trends["1year"])
        return {
            "location": location,
            "currency": "INR",
            "period": period,
            "price_change": t["price_change"],
            "market_type": "Seller's Market",
            "forecast_next_12mo": "+6% to +10%",
            "home_loan_rates": {"SBI": "8.50% p.a.", "HDFC": "8.70% p.a."},
            "note": "Data is simulated. Connect 99acres/MagicBricks APIs for live data."
        }
    else:
        trends = {
            "3months": {"price_change": "+1.8%", "inventory_change": "-5%", "dom_change": "-2 days"},
            "6months": {"price_change": "+3.1%", "inventory_change": "-8%", "dom_change": "-4 days"},
            "1year":   {"price_change": "+5.4%", "inventory_change": "-12%", "dom_change": "-6 days"},
            "3years":  {"price_change": "+18.2%", "inventory_change": "-22%", "dom_change": "-14 days"},
        }
        t = trends.get(period, trends["1year"])
        return {
            "location": location,
            "currency": "USD",
            "period": period,
            "median_home_price": "$482,000",
            "price_change": t["price_change"],
            "median_days_on_market": 16,
            "inventory_change": t["inventory_change"],
            "market_type": "Seller's Market",
            "price_per_sqft": "$268",
            "forecast_next_12mo": "+3.5% to +5.0%",
            "hottest_segments": ["3BR single family", "2BR condos under $350k"],
            "note": "Data is simulated. Connect Zillow/Redfin APIs for live data."
        }


# ─── Tool Dispatcher ──────────────────────────────────────────────────────────

def execute_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "search_properties":
        result = search_properties(**tool_input)
    elif tool_name == "calculate_mortgage":
        result = calculate_mortgage(**tool_input)
    elif tool_name == "get_neighborhood_info":
        result = get_neighborhood_info(**tool_input)
    elif tool_name == "get_market_trends":
        result = get_market_trends(**tool_input)
    else:
        result = {"error": f"Unknown tool: {tool_name}"}
    return json.dumps(result, indent=2)


# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert real estate assistant with deep knowledge of both the Indian market (especially Mumbai) and the US market.

**India / Mumbai expertise:**
- Mumbai neighborhoods: Bandra, Juhu, Worli, Lower Parel, Powai, Andheri, Malad, Kandivali, Thane, Navi Mumbai
- Indian property terminology: BHK, carpet area vs built-up area, RERA, stamp duty, registration, society charges
- Indian home loans: SBI (~8.50%), HDFC (~8.70%), ICICI (~8.75%) — always quote INR rates
- Indian taxes: Section 24b (interest), Section 80C (principal), capital gains on property
- Indian developers: Lodha, Godrej Properties, Oberoi Realty, Prestige, Hiranandani, Shapoorji
- Indian portals: 99acres, MagicBricks, Housing.com, NoBroker

**US expertise:**
- Mortgage rates, PMI, escrow, HOA, MLS listings
- US neighborhoods and market trends

**Tools available:** property search, home loan/EMI calculator, neighborhood analysis, market trends.

**Guidelines:**
- Detect whether the user is asking about India/Mumbai or US and respond accordingly
- For Indian queries: use INR, mention BHK format, quote Indian home loan rates (~8.5-9.5%), mention RERA
- For Mumbai: proactively mention which neighborhood fits the user's budget and lifestyle
- Always use tools proactively — if someone mentions a location or budget, search properties or calculate EMI
- Explain carpet area vs built-up area when discussing Indian flats
- Mention stamp duty (5-6% in Maharashtra) and registration costs for Mumbai buyers
- Flag important considerations: RERA registration, OC/CC certificate, society dues, parking costs
- Be warm, professional, and specific — give exact numbers not vague ranges when possible
"""

# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(title="Real Estate AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (must be done after route definitions that share prefixes)
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)

# ─── Request schema ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    messages: List[Any] = []
    user_message: str


# ─── SSE helper ───────────────────────────────────────────────────────────────

def sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ─── Streaming chat generator ─────────────────────────────────────────────────

async def stream_chat(messages: list):
    """
    Async generator that yields SSE-formatted strings.
    Handles the full tool-call loop within a single SSE connection.
    """
    loop = asyncio.get_event_loop()

    try:
        while True:
            # We run the blocking SDK call in a thread pool so we don't block the event loop.
            # Collect all events synchronously inside the thread, then yield them.
            collected_events = []

            def run_stream():
                with client.messages.stream(
                    model="claude-opus-4-6",
                    max_tokens=16000,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages,
                    thinking={"type": "adaptive"},
                ) as stream:
                    for event in stream:
                        collected_events.append(event)
                    return stream.get_final_message()

            final_message = await loop.run_in_executor(None, run_stream)

            # Process and yield collected events
            for event in collected_events:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta" and event.delta.text:
                        yield sse_event({"type": "text", "content": event.delta.text})
                        # Small sleep to allow the event loop to flush the response
                        await asyncio.sleep(0)

            # Append assistant turn to history
            messages.append({"role": "assistant", "content": final_message.content})

            # No tool calls — we are done
            if final_message.stop_reason != "tool_use":
                break

            # There are tool calls: execute them and notify the client
            tool_results = []
            for block in final_message.content:
                if block.type == "tool_use":
                    yield sse_event({
                        "type": "tool_call",
                        "name": block.name,
                        "input": block.input
                    })
                    await asyncio.sleep(0)

                    # Execute tool synchronously (all are pure Python / CPU-bound)
                    result_str = await loop.run_in_executor(
                        None, execute_tool, block.name, block.input
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str
                    })

            # Feed tool results back and loop
            messages.append({"role": "user", "content": tool_results})

        yield sse_event({"type": "done"})

    except Exception as exc:
        yield sse_event({"type": "error", "message": str(exc)})


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    return FileResponse(index_path)


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    # Build the messages list: history + new user message
    messages = list(request.messages)
    messages.append({"role": "user", "content": request.user_message})

    return StreamingResponse(
        stream_chat(messages),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# Mount static files last so explicit routes take precedence
app.mount("/static", StaticFiles(directory=static_dir), name="static")
