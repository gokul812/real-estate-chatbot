"""
Real Estate Chatbot powered by Claude claude-opus-4-6
Features: property search, mortgage calc, neighborhood info, market trends
"""

import anthropic
import json
import re

client = anthropic.Anthropic()

# ─── Tool Definitions ─────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "search_properties",
        "description": (
            "Search for properties based on location, price range, bedrooms, "
            "bathrooms, and property type. Returns a list of matching listings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City, neighborhood, or zip code"
                },
                "min_price": {
                    "type": "number",
                    "description": "Minimum price in USD"
                },
                "max_price": {
                    "type": "number",
                    "description": "Maximum price in USD"
                },
                "bedrooms": {
                    "type": "integer",
                    "description": "Number of bedrooms (minimum)"
                },
                "bathrooms": {
                    "type": "number",
                    "description": "Number of bathrooms (minimum)"
                },
                "property_type": {
                    "type": "string",
                    "enum": ["house", "condo", "townhouse", "apartment", "land", "any"],
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
        "name": "calculate_mortgage",
        "description": (
            "Calculate monthly mortgage payments, total interest paid, and "
            "amortization details given loan parameters."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "home_price": {
                    "type": "number",
                    "description": "Total home price in USD"
                },
                "down_payment_percent": {
                    "type": "number",
                    "description": "Down payment as a percentage (e.g. 20 for 20%)"
                },
                "annual_interest_rate": {
                    "type": "number",
                    "description": "Annual interest rate as a percentage (e.g. 6.5 for 6.5%)"
                },
                "loan_term_years": {
                    "type": "integer",
                    "description": "Loan term in years (typically 15 or 30)"
                },
                "property_tax_annual": {
                    "type": "number",
                    "description": "Annual property tax in USD (optional)"
                },
                "insurance_annual": {
                    "type": "number",
                    "description": "Annual homeowner's insurance in USD (optional)"
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

def search_properties(location, min_price=None, max_price=None, bedrooms=None,
                      bathrooms=None, property_type="any", for_sale_or_rent="sale"):
    """Simulated property search — replace with real MLS/API integration."""
    listings = [
        {
            "id": "PROP-001",
            "address": f"142 Oak Street, {location}",
            "price": 485000,
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
            "id": "PROP-002",
            "address": f"88 Maple Ave Unit 4B, {location}",
            "price": 320000,
            "bedrooms": 2,
            "bathrooms": 2,
            "sqft": 1100,
            "type": "condo",
            "status": "for sale",
            "days_on_market": 5,
            "description": "Modern condo in the heart of downtown with city views and rooftop access.",
            "features": ["city views", "rooftop deck", "in-unit laundry", "gym"]
        },
        {
            "id": "PROP-003",
            "address": f"310 Riverside Drive, {location}",
            "price": 725000,
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
            "id": "PROP-004",
            "address": f"55 Pine Street, {location}",
            "price": 2200,  # monthly rent
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
        if property_type != "any" and prop["type"] != property_type:
            continue
        results.append(prop)

    return {
        "location": location,
        "total_results": len(results),
        "listings": results if results else listings[:2],  # fallback so demo always shows data
        "note": "Results are simulated for demonstration. Connect a real MLS API for live data."
    }


def calculate_mortgage(home_price, down_payment_percent, annual_interest_rate,
                        loan_term_years, property_tax_annual=None, insurance_annual=None):
    """Accurate mortgage calculator."""
    down_payment = home_price * (down_payment_percent / 100)
    loan_amount = home_price - down_payment
    monthly_rate = (annual_interest_rate / 100) / 12
    num_payments = loan_term_years * 12

    if monthly_rate == 0:
        monthly_principal_interest = loan_amount / num_payments
    else:
        monthly_principal_interest = loan_amount * (
            monthly_rate * (1 + monthly_rate) ** num_payments
        ) / ((1 + monthly_rate) ** num_payments - 1)

    total_paid = monthly_principal_interest * num_payments
    total_interest = total_paid - loan_amount

    monthly_tax = (property_tax_annual / 12) if property_tax_annual else 0
    monthly_insurance = (insurance_annual / 12) if insurance_annual else 0
    total_monthly = monthly_principal_interest + monthly_tax + monthly_insurance

    pmi = 0
    if down_payment_percent < 20:
        pmi = loan_amount * 0.005 / 12  # ~0.5% annual PMI
        total_monthly += pmi

    return {
        "home_price": home_price,
        "down_payment": round(down_payment, 2),
        "down_payment_percent": down_payment_percent,
        "loan_amount": round(loan_amount, 2),
        "interest_rate": annual_interest_rate,
        "loan_term_years": loan_term_years,
        "monthly_principal_interest": round(monthly_principal_interest, 2),
        "monthly_property_tax": round(monthly_tax, 2),
        "monthly_insurance": round(monthly_insurance, 2),
        "monthly_pmi": round(pmi, 2) if pmi else None,
        "total_monthly_payment": round(total_monthly, 2),
        "total_interest_paid": round(total_interest, 2),
        "total_cost_of_loan": round(total_paid, 2),
        "pmi_note": "PMI applies because down payment is under 20%." if pmi else None
    }


def get_neighborhood_info(neighborhood, city, state=None):
    """Simulated neighborhood data — replace with real data source."""
    location_str = f"{neighborhood}, {city}" + (f", {state}" if state else "")
    return {
        "location": location_str,
        "overall_score": 82,
        "schools": {
            "rating": "8/10",
            "nearby": [
                {"name": f"{neighborhood} Elementary", "rating": "9/10", "distance": "0.3 mi"},
                {"name": f"{city} Middle School", "rating": "8/10", "distance": "0.7 mi"},
                {"name": f"{city} High School", "rating": "7/10", "distance": "1.2 mi"}
            ]
        },
        "safety": {
            "crime_rate": "below average",
            "safety_score": "7.8/10",
            "note": "Lower crime rate than 68% of U.S. neighborhoods"
        },
        "walkability": {
            "walk_score": 74,
            "bike_score": 62,
            "transit_score": 55,
            "rating": "Very Walkable"
        },
        "nearby_amenities": {
            "grocery_stores": 4,
            "restaurants": 23,
            "parks": 3,
            "hospitals": 1,
            "gyms": 5,
            "coffee_shops": 8
        },
        "real_estate": {
            "median_home_price": "$465,000",
            "median_rent_2br": "$1,950/mo",
            "price_trend_1yr": "+4.2%",
            "avg_days_on_market": 18
        },
        "demographics": {
            "median_household_income": "$72,400",
            "median_age": 34,
            "homeownership_rate": "58%"
        },
        "note": "Data is simulated for demonstration purposes."
    }


def get_market_trends(location, period="1year"):
    """Simulated market trend data."""
    trend_data = {
        "3months": {"price_change": "+1.8%", "inventory_change": "-5%", "dom_change": "-2 days"},
        "6months": {"price_change": "+3.1%", "inventory_change": "-8%", "dom_change": "-4 days"},
        "1year":   {"price_change": "+5.4%", "inventory_change": "-12%", "dom_change": "-6 days"},
        "3years":  {"price_change": "+18.2%", "inventory_change": "-22%", "dom_change": "-14 days"},
    }
    t = trend_data.get(period, trend_data["1year"])
    return {
        "location": location,
        "period": period,
        "median_home_price": "$482,000",
        "price_change": t["price_change"],
        "median_days_on_market": 16,
        "days_on_market_change": t["dom_change"],
        "active_listings": 342,
        "inventory_change": t["inventory_change"],
        "months_of_supply": 1.8,
        "market_type": "Seller's Market",
        "price_per_sqft": "$268",
        "forecast_next_12mo": "+3.5% to +5.0%",
        "hottest_segments": ["3BR single family", "2BR condos under $350k"],
        "note": "Data is simulated for demonstration. Connect Zillow/Redfin APIs for live data."
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

SYSTEM_PROMPT = """You are an expert real estate assistant with deep knowledge of:
- Property buying, selling, and renting
- Mortgage financing and loan options
- Neighborhood analysis and market trends
- Investment property evaluation
- First-time homebuyer guidance
- Real estate terminology and processes

You have access to tools to search listings, calculate mortgages, analyze neighborhoods, and retrieve market trends. Use them proactively when the user asks about properties, costs, or areas.

Guidelines:
- Be warm, professional, and knowledgeable
- Proactively use tools when relevant (e.g., if someone mentions a price/location, offer to calculate mortgage or search listings)
- Explain complex concepts clearly (e.g., amortization, escrow, cap rates)
- Always clarify whether data is real-time or simulated
- Ask clarifying questions to better understand user needs
- For mortgage calculations, always ask about interest rate if not provided and suggest current market rates (~6.5-7%)
- Flag important considerations (PMI, closing costs, inspection, etc.)
"""

# ─── Chat Loop ────────────────────────────────────────────────────────────────

def chat(messages: list) -> str:
    """Send messages, handle tool calls, return final assistant text."""
    while True:
        print("\n\033[90mAssistant:\033[0m ", end="", flush=True)

        # Stream the response
        full_content = []
        current_text = ""

        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
            thinking={"type": "adaptive"},
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        print(event.delta.text, end="", flush=True)
                        current_text += event.delta.text

            final = stream.get_final_message()

        print()  # newline after streamed output

        # Collect all content blocks
        full_content = final.content

        # Append assistant message to history
        messages.append({"role": "assistant", "content": full_content})

        # Check if we need to execute tools
        if final.stop_reason != "tool_use":
            break

        # Execute tools and collect results
        tool_results = []
        for block in full_content:
            if block.type == "tool_use":
                print(f"\n\033[33m[Tool: {block.name}({json.dumps(block.input, separators=(',',':'))})]\033[0m")
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

        messages.append({"role": "user", "content": tool_results})

    # Return final text
    for block in full_content:
        if hasattr(block, "text"):
            return block.text
    return ""


def main():
    print("\033[36m" + "=" * 60)
    print("  Real Estate AI Assistant")
    print("=" * 60 + "\033[0m")
    print("Ask me anything about buying, selling, renting, mortgages,")
    print("neighborhoods, or market trends. Type 'quit' to exit.\n")

    messages = []

    while True:
        try:
            user_input = input("\033[32mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! Happy house hunting!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("Goodbye! Happy house hunting!")
            break

        messages.append({"role": "user", "content": user_input})
        chat(messages)


if __name__ == "__main__":
    main()
