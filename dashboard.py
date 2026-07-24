# Business Performance Dashboard
# Run with: streamlit run dashboard.py
# (NOT python3 dashboard.py -- Streamlit apps need the streamlit command
# so it can spin up the local web server and re-run this script on every interaction)

import datetime
import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from groq import Groq

# Keys live only in .env (gitignored) -- load_dotenv() reads them into the
# process environment; nothing here ever hardcodes or prints a key.
load_dotenv()

# st.set_page_config MUST be the first Streamlit command in the script.
# layout="wide" gives us the full browser width instead of a narrow centered column --
# important once we add side-by-side KPI cards and charts.
st.set_page_config(page_title="Business Performance Dashboard", layout="wide")

st.title("Business Performance Dashboard")

# @st.cache_data is Streamlit-specific: Streamlit reruns your ENTIRE script top-to-bottom
# every time a user touches a widget (a filter, a dropdown, etc). Without caching, that
# means re-reading and re-parsing the CSV from disk on every single click.
# This decorator tells Streamlit: "run this function once, remember the result, and
# reuse it as long as the CSV file/args haven't changed."
@st.cache_data
def load_data():
    df = pd.read_csv("superstore.csv")
    # This CSV has extra rows tacked on below the real order data (looks like a
    # "Returns"/"People" sheet from the original Excel workbook got appended to the
    # "Orders" sheet during export). Those junk rows have non-numeric Row ID values
    # ("Yes", a person's name, etc) and blank Order Date/Region/etc. Keep only rows
    # where Row ID is a real number.
    df = df[pd.to_numeric(df["Row ID"], errors="coerce").notna()].copy()
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    # Precompute this here, inside the cached function, not later in the script --
    # anything inside load_data() only runs once and gets reused on every rerun.
    # If we computed this after calling load_data(), it would recalculate on every
    # single filter click since it'd be outside the cache boundary.
    df["Order Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
    return df

df = load_data()

# --- Sidebar filters ---
# st.sidebar puts widgets in the left panel instead of the main page.
# Each widget call returns the user's current selection immediately -- no event
# handlers, no callbacks. Streamlit re-runs this whole script after every change,
# so "region" below is just a plain list of strings by the time we reach this line.
st.sidebar.header("Filters")

regions = sorted(df["Region"].unique())
selected_regions = st.sidebar.multiselect("Region", regions, default=regions)

categories = sorted(df["Category"].unique())
selected_categories = st.sidebar.multiselect("Category", categories, default=categories)

min_date = df["Order Date"].min().date()
max_date = df["Order Date"].max().date()
# date_input with a (start, end) default returns a tuple of two dates once the user
# has picked both ends of the range. While they've only picked one, it returns a
# single date -- that's why we guard for length below instead of unpacking directly.
date_range = st.sidebar.date_input("Order Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

# Apply all three filters together. This filtered_df is what every KPI, chart,
# and insight downstream should read from -- never the original df.
filtered_df = df[
    df["Region"].isin(selected_regions)
    & df["Category"].isin(selected_categories)
    & (df["Order Date"].dt.date >= start_date)
    & (df["Order Date"].dt.date <= end_date)
]

row1, row2 = st.columns([3, 1])
row1.write(f"Showing {len(filtered_df):,} of {len(df):,} rows after filters.")
row2.download_button(
    ":material/download: Download CSV",
    data=filtered_df.to_csv(index=False),
    file_name="superstore_filtered.csv",
    mime="text/csv",
    use_container_width=True,
)

# --- KPI cards ---
# st.columns(4) splits the page into 4 equal-width slots and returns one "container"
# object per slot. Writing to col1/col2/etc (via `with col1:` or col1.metric(...))
# places content side by side instead of stacked top-to-bottom like normal Streamlit
# calls. This is the standard way to lay out a KPI row.
total_revenue = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
total_orders = filtered_df["Order ID"].nunique()
profit_margin = (total_profit / total_revenue * 100) if total_revenue else 0

# Each card also compares the current filtered period to the immediately
# preceding period of the same length (same region/category filters) so the
# cards show momentum, not just a snapshot. If the date filter already covers
# the full dataset there's nothing earlier to compare against -- prev_*_df
# comes back empty and the deltas simply don't show, rather than displaying a
# misleading number.
period_length_days = (end_date - start_date).days + 1
prev_end_date = start_date - datetime.timedelta(days=1)
prev_start_date = prev_end_date - datetime.timedelta(days=period_length_days - 1)
prev_period_df = df[
    df["Region"].isin(selected_regions)
    & df["Category"].isin(selected_categories)
    & (df["Order Date"].dt.date >= prev_start_date)
    & (df["Order Date"].dt.date <= prev_end_date)
]
prev_revenue = prev_period_df["Sales"].sum()
prev_profit = prev_period_df["Profit"].sum()
prev_orders = prev_period_df["Order ID"].nunique()
prev_margin = (prev_profit / prev_revenue * 100) if prev_revenue else 0


def pct_delta(current, previous):
    if not previous:
        return None
    return (current - previous) / previous * 100


revenue_delta = pct_delta(total_revenue, prev_revenue)
profit_delta = pct_delta(total_profit, prev_profit)
orders_delta = pct_delta(total_orders, prev_orders)
margin_delta = (profit_margin - prev_margin) if prev_revenue else None  # percentage points, not %-of-%

col1, col2, col3, col4 = st.columns(4)
# st.metric renders a label, a big number, and an optional delta line below it --
# border=True gives the card outline natively, and delta_description adds the
# small muted "vs prior period" caption next to the colored delta arrow.
col1.metric(
    ":material/payments: Revenue",
    f"${total_revenue:,.0f}",
    delta=f"{revenue_delta:+.1f}%" if revenue_delta is not None else None,
    delta_description="vs prior period" if revenue_delta is not None else None,
    border=True,
)
col2.metric(
    ":material/trending_up: Profit",
    f"${total_profit:,.0f}",
    delta=f"{profit_delta:+.1f}%" if profit_delta is not None else None,
    delta_description="vs prior period" if profit_delta is not None else None,
    border=True,
)
col3.metric(
    ":material/receipt_long: Orders",
    f"{total_orders:,}",
    delta=f"{orders_delta:+.1f}%" if orders_delta is not None else None,
    delta_description="vs prior period" if orders_delta is not None else None,
    border=True,
)
col4.metric(
    ":material/percent: Profit Margin",
    f"{profit_margin:.1f}%",
    delta=f"{margin_delta:+.1f} pts" if margin_delta is not None else None,
    delta_description="vs prior period" if margin_delta is not None else None,
    border=True,
)

st.subheader("Top 5 Sales by Profit")
# Keep only the columns that matter for this table -- the full row has 21 columns
# (postal code, ship mode, etc.) that just bury the Profit figure we care about.
top_by_profit = (
    filtered_df.sort_values("Profit", ascending=False)
    .head(5)[["Product Name", "Category", "Region", "Sales", "Profit"]]
    .reset_index(drop=True)
)
top_by_profit.index = top_by_profit.index + 1  # start ranking at 1, not 0
top_by_profit.index.name = "Rank"

# column_config lets us format specific columns without touching the underlying data --
# here Sales/Profit render as currency instead of raw floats. hide_index=True would drop
# our Rank column too, so instead we've named the index "Rank" above and left it visible.
st.dataframe(
    top_by_profit,
    column_config={
        "Sales": st.column_config.NumberColumn("Sales", format="$%.2f"),
        "Profit": st.column_config.NumberColumn("Profit", format="$%.2f"),
    },
    use_container_width=True,
)

# --- Charts ---
# st.plotly_chart renders a Plotly figure inline. use_container_width=True stretches
# it to fill its column instead of Plotly's default fixed pixel width -- matters here
# since we're placing charts side by side in columns.
st.subheader("Charts")

# Fixed region -> color mapping (alphabetical order, not by current selection
# or by revenue rank) so a color always means the same region even as the
# region filter changes -- deselecting South must not repaint Central. Hexes
# are slots 1-4 of the validated categorical palette (blue/orange/aqua/yellow),
# which clears the colorblind-safe adjacency checks for line charts.
REGION_COLORS_LIGHT = {"Central": "#2a78d6", "East": "#eb6834", "South": "#1baf7a", "West": "#eda100"}
REGION_COLORS_DARK = {"Central": "#3987e5", "East": "#d95926", "South": "#199e70", "West": "#c98500"}

# st.context.theme.type reflects the viewer's actual active theme (their
# toggle, not just the app's config default), so the chart matches whichever
# mode they're looking at.
is_dark_theme = st.context.theme.type == "dark"
region_colors = REGION_COLORS_DARK if is_dark_theme else REGION_COLORS_LIGHT
single_line_color = "#3987e5" if is_dark_theme else "#2a78d6"
grid_color = "#2c2c2a" if is_dark_theme else "#e1e0d9"
axis_line_color = "#383835" if is_dark_theme else "#c3c2b7"
muted_ink = "#898781"
secondary_ink = "#c3c2b7" if is_dark_theme else "#52514e"


def style_trend_chart(fig, show_legend):
    # Bigger/clearer axis text, recessive gridlines (horizontal only -- vertical
    # gridlines on a monthly time series just add noise), and a compact legend
    # anchored above the plot so it doesn't cover any lines.
    fig.update_traces(line=dict(width=2), marker=dict(size=8))
    fig.update_layout(
        font=dict(size=13, color=secondary_ink),
        title=dict(font=dict(size=18)),
        xaxis=dict(
            title=dict(text="Month", font=dict(size=15)),
            tickfont=dict(size=13, color=muted_ink),
            showgrid=False,
            showline=True,
            linecolor=axis_line_color,
        ),
        yaxis=dict(
            title=dict(text="Sales", font=dict(size=15)),
            tickfont=dict(size=13, color=muted_ink),
            tickprefix="$",
            tickformat=",.0f",
            showgrid=True,
            gridcolor=grid_color,
            gridwidth=1,
            zeroline=False,
            showline=False,
        ),
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=13), title=None),
        hovermode="x unified",
        margin=dict(t=60),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


compare_by_region = st.checkbox("Compare by Region", help="Show a separate sales line per region instead of one combined line.")

if compare_by_region:
    monthly_by_region = filtered_df.groupby(["Order Month", "Region"], as_index=False)["Sales"].sum()
    fig_trend = px.line(
        monthly_by_region,
        x="Order Month",
        y="Sales",
        color="Region",
        color_discrete_map=region_colors,
        # Pins line/legend order to the fixed alphabetical region order (matching
        # the fixed color assignment above) instead of whatever order the data
        # happens to produce.
        category_orders={"Region": sorted(region_colors.keys())},
        markers=True,
        title="Monthly Sales Trend by Region",
    )
    fig_trend = style_trend_chart(fig_trend, show_legend=True)
else:
    monthly = filtered_df.groupby("Order Month", as_index=False)["Sales"].sum()
    fig_trend = px.line(monthly, x="Order Month", y="Sales", markers=True, title="Monthly Sales Trend")
    fig_trend.update_traces(line_color=single_line_color, marker_color=single_line_color)
    fig_trend = style_trend_chart(fig_trend, show_legend=False)

st.plotly_chart(fig_trend, use_container_width=True)

def style_bar_chart(fig, y_title):
    # Same visual language as the trend chart above (recessive gridlines,
    # bigger axis text, transparent background) so the whole Charts section
    # reads as one consistent system rather than a mix of default Plotly
    # styling. Single-series bars, so no legend needed -- the axis labels
    # already identify each bar.
    fig.update_traces(marker_color=single_line_color)
    fig.update_layout(
        font=dict(size=13, color=secondary_ink),
        title=dict(font=dict(size=18)),
        xaxis=dict(title=dict(font=dict(size=15)), tickfont=dict(size=13, color=muted_ink), showgrid=False, showline=True, linecolor=axis_line_color),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=15)),
            tickfont=dict(size=13, color=muted_ink),
            tickprefix="$",
            tickformat=",.0f",
            showgrid=True,
            gridcolor=grid_color,
            gridwidth=1,
            zeroline=False,
            showline=False,
        ),
        showlegend=False,
        margin=dict(t=60),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


chart_col1, chart_col2, chart_col3 = st.columns(3)

revenue_by_category = filtered_df.groupby("Category", as_index=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum")).sort_values("Sales", ascending=False)
fig_category = px.bar(revenue_by_category, x="Category", y="Sales", title="Revenue by Category")
chart_col1.plotly_chart(style_bar_chart(fig_category, "Sales"), use_container_width=True)

profit_by_region = filtered_df.groupby("Region", as_index=False)["Profit"].sum().sort_values("Profit", ascending=False)
fig_region = px.bar(profit_by_region, x="Region", y="Profit", title="Profit by Region")
chart_col2.plotly_chart(style_bar_chart(fig_region, "Profit"), use_container_width=True)

revenue_by_segment = filtered_df.groupby("Segment", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
fig_segment = px.bar(revenue_by_segment, x="Segment", y="Sales", title="Revenue by Segment")
chart_col3.plotly_chart(style_bar_chart(fig_segment, "Sales"), use_container_width=True)

# --- Top Performers ---
# Aggregated by Product/Customer (summed across every line item), unlike the
# "Top 5 Sales by Profit" table above which ranks individual order lines --
# this answers "which products/customers matter most overall" instead.
st.subheader("Top Performers")
perf_col1, perf_col2 = st.columns(2)

top_products = (
    filtered_df.groupby("Product Name", as_index=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum")).sort_values("Sales", ascending=False).head(5).reset_index(drop=True)
)
top_products.index += 1
top_products.index.name = "Rank"

top_customers = (
    filtered_df.groupby("Customer Name", as_index=False)
    .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
    .sort_values("Sales", ascending=False)
    .head(5)
    .reset_index(drop=True)
)
top_customers.index += 1
top_customers.index.name = "Rank"

currency_column_config = {
    "Sales": st.column_config.NumberColumn("Sales", format="$%.2f"),
    "Profit": st.column_config.NumberColumn("Profit", format="$%.2f"),
}

with perf_col1:
    st.markdown("**Top 5 Products by Revenue**")
    st.dataframe(top_products, column_config=currency_column_config, use_container_width=True)

with perf_col2:
    st.markdown("**Top 5 Customers by Revenue**")
    st.dataframe(top_customers, column_config=currency_column_config, use_container_width=True)

# --- AI-generated insights ---
# Instead of hand-written sentence templates, we hand a real LLM a list of
# already-computed, already-formatted facts for the current filter selection
# (never the raw CSV/full row data) and ask it to return a small JSON list of
# {headline, detail} insights -- rendered below as individual cards. Groq is
# tried first (fast/cheap); if it errors or returns something unusable, we
# fall back to Gemini. Both API keys are read from the environment (populated
# by load_dotenv() above) and are never hardcoded or displayed in the UI.
st.subheader("AI-Generated Insights")


def fmt_money(x):
    # One decimal place, comma thousands separators, sign-aware -- this is the
    # single source of truth for currency formatting so every card (and every
    # number an LLM copies verbatim into a sentence) looks the same.
    return f"-${abs(x):,.1f}" if x < 0 else f"${x:,.1f}"


def fmt_pct(x):
    return f"{x:.1f}%"


def strip_markdown(text):
    # Safety net alongside the prompt's "plain text only" rule -- these cards
    # are rendered with st.markdown/st.write, so stray markdown syntax from the
    # model renders as real formatting instead of plain text. Backticks/
    # asterisks/underscores would turn into code spans or bold/italic runs.
    for char in ("`", "*", "_"):
        text = text.replace(char, "")
    # Every insight sentence has at least one "$1,234.5" figure, often two --
    # and Streamlit's markdown renderer treats a pair of "$" as LaTeX math
    # delimiters, silently rendering everything between them (dollar signs and
    # all) as an equation instead of plain text. Escaping "$" is what stops that.
    text = text.replace("$", r"\$")
    return text


if filtered_df.empty:
    st.warning("No data matches the current filters.")
else:
    top_category = revenue_by_category.iloc[0]
    worst_category = revenue_by_category.iloc[-1]
    strongest_region = profit_by_region.iloc[0]
    weakest_region = profit_by_region.iloc[-1]
    num_categories_selected = len(revenue_by_category)
    num_regions_selected = len(profit_by_region)
    top_category_margin = (top_category["Profit"] / top_category["Sales"] * 100) if top_category["Sales"] else 0
    worst_category_margin = (worst_category["Profit"] / worst_category["Sales"] * 100) if worst_category["Sales"] else 0

    # A list of (label, formatted value) facts, not raw numbers -- the LLM is
    # told to copy these values character-for-character rather than format
    # numbers itself, which is what caused the inconsistent $/％/decimal
    # formatting in the old free-text version. Region/category *comparison*
    # facts are only included when there's actually more than one to compare,
    # so a narrow filter (e.g. one region selected) can't produce a nonsensical
    # "X is both the best and worst" insight. Customer/product facts reuse the
    # Top Performers tables computed above -- they're what let the model surface
    # something genuinely non-obvious (a top-revenue customer or product that's
    # actually thin/negative on margin) instead of just re-describing the charts.
    facts = [
        {"label": "Total Revenue", "value": fmt_money(total_revenue)},
        {"label": "Total Profit", "value": fmt_money(total_profit)},
        {"label": "Overall Profit Margin", "value": fmt_pct(profit_margin)},
        {
            "label": "Top Category by Revenue",
            "value": f"{top_category['Category']} ({fmt_money(top_category['Sales'])}, "
            f"{fmt_pct(top_category['Sales'] / total_revenue * 100 if total_revenue else 0)} of total revenue, {fmt_pct(top_category_margin)} margin)",
        },
        {"label": "Most Profitable Region", "value": f"{strongest_region['Region']} ({fmt_money(strongest_region['Profit'])} profit)"},
    ]
    if num_categories_selected > 1:
        facts.append({"label": "Lowest-Revenue Category", "value": f"{worst_category['Category']} ({fmt_money(worst_category['Sales'])}, {fmt_pct(worst_category_margin)} margin)"})
        facts.append({"label": "Revenue Gap: Top vs Lowest Category", "value": fmt_money(top_category["Sales"] - worst_category["Sales"])})
    if num_regions_selected > 1:
        facts.append({"label": "Least Profitable Region", "value": f"{weakest_region['Region']} ({fmt_money(weakest_region['Profit'])} profit)"})
        facts.append({"label": "Profit Gap: Best vs Worst Region", "value": fmt_money(strongest_region["Profit"] - weakest_region["Profit"])})
    if not top_customers.empty:
        top_customer = top_customers.iloc[0]
        top_customer_margin = (top_customer["Profit"] / top_customer["Sales"] * 100) if top_customer["Sales"] else 0
        facts.append(
            {"label": "Top Customer by Revenue", "value": f"{top_customer['Customer Name']} ({fmt_money(top_customer['Sales'])} revenue, {fmt_money(top_customer['Profit'])} profit, {fmt_pct(top_customer_margin)} margin)"}
        )
        unprofitable_top_customers = top_customers[top_customers["Profit"] < 0]
        if not unprofitable_top_customers.empty:
            worst_customer = unprofitable_top_customers.sort_values("Profit").iloc[0]
            facts.append(
                {"label": "Unprofitable Top-5 Customer (by revenue)", "value": f"{worst_customer['Customer Name']} ({fmt_money(worst_customer['Sales'])} revenue but {fmt_money(worst_customer['Profit'])} profit)"}
            )
    if not top_products.empty:
        top_product = top_products.iloc[0]
        top_product_margin = (top_product["Profit"] / top_product["Sales"] * 100) if top_product["Sales"] else 0
        facts.append(
            {"label": "Top Product by Revenue", "value": f"{top_product['Product Name']} ({fmt_money(top_product['Sales'])} revenue, {fmt_money(top_product['Profit'])} profit, {fmt_pct(top_product_margin)} margin)"}
        )

    # Sort keys so the JSON string is stable for a given set of facts -- this
    # string is the cache key below, so identical filtered data (even from a
    # different combination of filters) reuses the cached LLM call instead of
    # re-hitting the API on every rerun.
    facts_json = json.dumps(facts, sort_keys=True)

    PROMPT_TEMPLATE = """You are a sharp business advisor writing a short insights panel for a dashboard. The reader glances at this for a few seconds -- every insight has to earn its place by telling them something to DO, not something they could already see on the chart next to it.

Using ONLY the facts given below, write 3 to 4 advice-driven insights as JSON. Quality over quantity: 3 sharp insights beat 4 where one is filler.

Formatting rule (critical): every dollar figure and percentage you mention MUST be copied character-for-character from the "value" fields below (same $, same commas, same one decimal place). Never round, abbreviate, truncate, or recompute a number yourself.

Output shape: {{"insights": [{{"headline": "...", "detail": "..."}}, ...]}}
- "headline": 2-5 words, a specific action, not a topic label.
- "detail": ONE natural sentence, 18-30 words: name the specific action, then back it with the number(s) that justify it. Write like a sharp advisor talking, not a report reciting stats.
- Prioritize the most surprising or non-obvious fact available (e.g. a customer or product that looks strong on revenue but is weak on margin/profit) over a generic "grow the leader" statement -- the reader can already see category/region totals on the chart, so don't just repeat those unless there's a real gap or margin problem attached.
- Ban vague corporate filler with no concrete next step: phrases like "maximize potential", "significant opportunity", "moving forward", "drive growth", "leverage strengths" are not allowed. Every sentence needs one concrete, specific action a person could actually go do (e.g. "review the discount terms on this account", "run a price test", "audit the cost breakdown", "shift ad spend toward X") -- not an abstract goal.
- Ground every recommendation only in the numbers given -- never invent a cause not in the data (e.g. don't claim to know WHY a margin is low, just flag it and recommend investigating or acting on it).
- Use each fact's label only to understand what the value means -- never print a label or field name verbatim, write it as natural English instead.
- Plain text only: no markdown, no backticks, no bold/italic asterisks, no code formatting of any kind around numbers or words.

Example of the quality bar (unrelated dataset -- do not reuse this wording or these numbers):
{{"headline": "Audit Top Account Discounts", "detail": "Your highest-revenue customer, Dana Reyes, generates $61,400.0 in sales but only $-2,150.0 in profit, so review the discounting on that account before courting more volume there."}}

Facts:
{facts}
"""

    def parse_insights(raw_json: str):
        data = json.loads(raw_json)
        items = data["insights"]
        cleaned = [
            {"headline": strip_markdown(str(item["headline"]).strip()), "detail": strip_markdown(str(item["detail"]).strip())}
            for item in items
            if isinstance(item, dict) and item.get("headline") and item.get("detail")
        ]
        if not cleaned:
            raise ValueError("Model returned no usable insights.")
        return cleaned[:4]

    @st.cache_data(show_spinner=False)
    def generate_ai_insights(facts_json: str):
        prompt = PROMPT_TEMPLATE.format(facts=facts_json)
        errors = []

        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                client = Groq(api_key=groq_key)
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_tokens=700,
                    response_format={"type": "json_object"},
                )
                return parse_insights(response.choices[0].message.content)
            except Exception as e:
                errors.append(f"Groq: {e}")

        gemini_key = os.getenv("GOOGLE_API_KEY")
        if gemini_key:
            try:
                client = genai.Client(api_key=gemini_key)
                response = client.models.generate_content(
                    model="gemini-flash-latest",
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(response_mime_type="application/json"),
                )
                return parse_insights(response.text)
            except Exception as e:
                errors.append(f"Gemini: {e}")

        raise RuntimeError("No AI provider succeeded. " + "; ".join(errors) if errors else "No GROQ_API_KEY or GOOGLE_API_KEY found in .env.")

    try:
        with st.spinner("Generating insights..."):
            ai_insights = generate_ai_insights(facts_json)
        cols = st.columns(len(ai_insights))
        for col, insight in zip(cols, ai_insights):
            with col:
                with st.container(border=True):
                    st.markdown(f"**:material/lightbulb: {insight['headline']}**")
                    st.write(insight["detail"])
    except Exception as e:
        st.error(f"Couldn't generate AI insights right now: {e}")
