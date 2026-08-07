# Performance Dashboards
# Run with: streamlit run app.py (or ./run.sh)
# (NOT python3 app.py -- Streamlit apps need the streamlit command
# so it can spin up the local web server and re-run this script on every interaction)

import datetime
import json
import os
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from groq import Groq
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Keys live only in .env (gitignored) -- load_dotenv() reads them into the
# process environment; nothing here ever hardcodes or prints a key.
load_dotenv()

# st.set_page_config MUST be the first Streamlit command in the script.
# layout="wide" gives us the full browser width instead of a narrow centered column --
# important once we add side-by-side KPI cards and charts.
st.set_page_config(page_title="Performance Dashboards", layout="wide")

st.title("Performance Dashboards")

# =============================================================================
# Shared helpers -- used by BOTH the Business Performance tab and the Stock
# Market tab, so formatting and the AI-insights plumbing stay consistent
# between them instead of two copies drifting apart.
# =============================================================================


def fmt_money(x):
    # One decimal place, comma thousands separators, sign-aware -- for the
    # Business tab's revenue/profit figures (hundreds to hundreds-of-thousands).
    return f"-${abs(x):,.1f}" if x < 0 else f"${x:,.1f}"


def fmt_price(x):
    # Two decimal places -- stock prices are quoted in cents, unlike the
    # Business tab's larger aggregate dollar figures.
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


def fmt_pct(x):
    # Unsigned -- for magnitudes/ratios (a margin, a share of total, volatility).
    # These aren't "changes", so a forced leading "+" would read oddly on a
    # plain 12.5% margin.
    return f"{x:.1f}%" if x is not None else "N/A"


def fmt_change_pct(x):
    # Signed -- for deltas (daily change, period change), where the +/- is the
    # whole point and drives st.metric's up/down arrow color.
    return f"{x:+.1f}%" if x is not None else "N/A"


def fmt_price_change(x):
    # Signed dollar delta, same spirit as fmt_change_pct but for a $ move
    # instead of a % move (e.g. "+$3.21" today).
    return f"-${abs(x):,.2f}" if x < 0 else f"+${x:,.2f}"


def strip_markdown(text):
    # Safety net alongside each prompt's "plain text only" rule -- these cards
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


def parse_insights(raw_json: str):
    # Shared validator for both tabs' LLM output: same {"insights": [{headline,
    # detail}, ...]} shape either way, so one parser covers both.
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


def call_llm_json(prompt: str, temperature: float = 0.5, max_tokens: int = 700) -> str:
    # The one place that actually talks to an LLM -- both tabs' cached
    # generate_*_insights() wrappers call this with their own prompt, so
    # "same API call pattern" is literally the same function, not two
    # hand-copied implementations that can drift apart. Groq is tried first
    # (fast/cheap); Gemini is the fallback if Groq errors or isn't configured.
    # Both keys come from the environment (populated by load_dotenv() above)
    # and are never hardcoded or displayed in the UI.
    errors = []

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            client = Groq(api_key=groq_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
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
            return response.text
        except Exception as e:
            errors.append(f"Gemini: {e}")

    raise RuntimeError("No AI provider succeeded. " + "; ".join(errors) if errors else "No GROQ_API_KEY or GOOGLE_API_KEY found in .env.")


# --- Shared chart theme ---
# st.context.theme.type reflects the viewer's actual active theme (their
# toggle, not just the app's config default), so every chart in both tabs
# matches whichever mode they're looking at.
is_dark_theme = st.context.theme.type == "dark"

# Slots 1-4 of a colorblind-validated categorical palette (blue/orange/aqua/
# yellow), fixed order, never cycled -- reused everywhere a chart needs
# distinct series colors (region lines, a price line vs its moving averages,
# two compared stocks).
CHART_PALETTE_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CHART_PALETTE_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500"]
chart_palette = CHART_PALETTE_DARK if is_dark_theme else CHART_PALETTE_LIGHT

REGION_COLORS_LIGHT = {"Central": "#2a78d6", "East": "#eb6834", "South": "#1baf7a", "West": "#eda100"}
REGION_COLORS_DARK = {"Central": "#3987e5", "East": "#d95926", "South": "#199e70", "West": "#c98500"}
region_colors = REGION_COLORS_DARK if is_dark_theme else REGION_COLORS_LIGHT

single_line_color = chart_palette[0]
grid_color = "#2c2c2a" if is_dark_theme else "#e1e0d9"
axis_line_color = "#383835" if is_dark_theme else "#c3c2b7"
muted_ink = "#898781"
secondary_ink = "#c3c2b7" if is_dark_theme else "#52514e"


def style_trend_chart(fig, show_legend, x_title="Month", y_title="Sales", y_tickformat=",.0f", y_tickprefix="$", y_ticksuffix="", y_zeroline=False):
    # Bigger/clearer axis text, recessive gridlines (horizontal only -- vertical
    # gridlines on a time series just add noise), and a compact legend anchored
    # above the plot so it doesn't cover any lines. Shared by every line chart
    # in both tabs; the y_* params are what let the same function format a
    # sales-dollar axis, a price-dollar axis, and a signed-percent axis.
    fig.update_traces(line=dict(width=2), marker=dict(size=8))
    fig.update_layout(
        font=dict(size=13, color=secondary_ink),
        title=dict(font=dict(size=18)),
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=15)),
            tickfont=dict(size=13, color=muted_ink),
            showgrid=False,
            showline=True,
            linecolor=axis_line_color,
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=15)),
            tickfont=dict(size=13, color=muted_ink),
            tickprefix=y_tickprefix,
            ticksuffix=y_ticksuffix,
            tickformat=y_tickformat,
            showgrid=True,
            gridcolor=grid_color,
            gridwidth=1,
            zeroline=y_zeroline,
            zerolinecolor=axis_line_color,
            zerolinewidth=1,
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


def style_bar_chart(fig, y_title, y_tickprefix="$", y_tickformat=",.0f"):
    # Same visual language as the trend chart above (recessive gridlines,
    # bigger axis text, transparent background) so the whole Charts section
    # reads as one consistent system rather than a mix of default Plotly
    # styling. Single-series bars, so no legend needed -- the axis labels
    # already identify each bar. y_tickprefix/y_tickformat default to the
    # Business tab's dollar bars but are overridable for a non-dollar bar
    # chart (e.g. a 0-1 feature-importance score).
    fig.update_traces(marker_color=single_line_color)
    fig.update_layout(
        font=dict(size=13, color=secondary_ink),
        title=dict(font=dict(size=18)),
        xaxis=dict(title=dict(font=dict(size=15)), tickfont=dict(size=13, color=muted_ink), showgrid=False, showline=True, linecolor=axis_line_color),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=15)),
            tickfont=dict(size=13, color=muted_ink),
            tickprefix=y_tickprefix,
            tickformat=y_tickformat,
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


# --- Shared AI-insight card ---
# Same {headline, detail} shape everywhere (Business, Stock), so one card
# renderer + one tone-coloring rule covers both instead of each section
# hand-rolling its own st.container/st.markdown pair.

# Slots into chart_palette -- no new hex values, just naming which existing
# slot each tone borrows: aqua/green for good news, blue (the app's default
# accent) for a plain informational insight, and amber/yellow -- the same
# hue family Streamlit's own st.warning uses -- for anything cautionary.
INSIGHT_TONE_PALETTE_SLOT = {"positive": 2, "neutral": 0, "caution": 3}


def infer_insight_tone(headline: str, detail: str) -> str:
    """Best-effort tone classification for card styling ONLY -- a lightweight
    keyword/pattern heuristic that runs AFTER parse_insights() has already
    returned clean {headline, detail} pairs. It never touches what's asked of
    the LLM, how its output is parsed, or what gets cached -- purely a
    rendering-time decision about which color left-border to draw. Good
    enough to pick an accent color, not meant to be a real sentiment model.
    """
    text = f"{headline} {detail}".lower()
    # A literal negative figure (e.g. "-$1,980.7" or "-7.9%") is a strong,
    # unambiguous cue on its own regardless of the words around it.
    has_negative_figure = bool(re.search(r"-\d", text))
    caution_words = (
        "unprofitable", "losing", "loss", "audit", "investigat", "review the", "declin", "risk", "warn",
        "weak", "underperform", "concern", "high volatility", "did not", "didn't", "gone quiet",
    )
    positive_words = (
        "leader", "leads", "top ", "strong", "grow", "increas", "outperform", "gain", "promote",
        "double down", "uptrend", "beat", "high-margin", "high margin", "calm",
    )
    if has_negative_figure or any(word in text for word in caution_words):
        return "caution"
    if any(word in text for word in positive_words):
        return "positive"
    return "neutral"


def render_insight_card(col, insight: dict, eyebrow: str, index: int):
    """Render one AI-generated insight as a tone-colored card. Still a real
    st.container(border=True) -- Streamlit doesn't expose a border-color
    param, so the colored left accent + tint comes from a tiny scoped
    <style> block targeting this container's key (`.st-key-<key>`, a
    stable class Streamlit generates from the `key=` argument), injected via
    st.html right before the container it styles. color-mix(...,transparent)
    keeps the tint subtle and correct on both themes without hardcoding two
    separate background colors.
    """
    tone = infer_insight_tone(insight["headline"], insight["detail"])
    tone_color = chart_palette[INSIGHT_TONE_PALETTE_SLOT[tone]]
    key = f"insight-{eyebrow.lower().replace(' ', '-')}-{index}"
    tint_pct = 14 if is_dark_theme else 8  # a flat tint reads lighter on a dark surface, so it gets a bit more

    with col:
        st.html(f"""<style>
.st-key-{key} {{
    border-left: 4px solid {tone_color} !important;
    background: color-mix(in srgb, {tone_color} {tint_pct}%, transparent) !important;
}}
</style>""")
        with st.container(border=True, key=key):
            st.caption(eyebrow.upper())
            st.markdown(f"##### :material/lightbulb: {insight['headline']}")
            st.write(insight["detail"])


# --- Shared Superstore loader ---
# @st.cache_data is Streamlit-specific: Streamlit reruns your ENTIRE script top-to-bottom
# every time a user touches a widget (a filter, a dropdown, etc). Without caching, that
# means re-reading and re-parsing the CSV from disk on every single click.
# This decorator tells Streamlit: "run this function once, remember the result, and
# reuse it as long as the CSV file/args haven't changed." Module-level (not nested inside
# render_business_dashboard) so the Churn Radar and Sales Forecast tabs can reuse the exact
# same cleaned data and precomputed "Order Month" column instead of re-reading the CSV.
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


# =============================================================================
# Business Performance Dashboard (tab 1)
# =============================================================================


def render_business_dashboard():
    st.header("Business Performance Dashboard")

    df = load_data()

    # --- Sidebar filters ---
    # st.sidebar puts widgets in the left panel instead of the main page.
    # Each widget call returns the user's current selection immediately -- no event
    # handlers, no callbacks. Streamlit re-runs this whole script after every change,
    # so "region" below is just a plain list of strings by the time we reach this line.
    st.sidebar.header("Filters")
    st.sidebar.caption("Applies to the Business Performance tab.")

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
    # {headline, detail} insights -- rendered below as individual cards.
    st.subheader("AI-Generated Insights")

    if filtered_df.empty:
        st.warning("No data matches the current filters.")
        return

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
    # numbers itself, which is what caused inconsistent $/%/decimal formatting
    # in an earlier free-text version. Region/category *comparison* facts are
    # only included when there's actually more than one to compare, so a
    # narrow filter (e.g. one region selected) can't produce a nonsensical "X
    # is both the best and worst" insight. Customer/product facts reuse the
    # Top Performers tables computed above -- they're what let the model
    # surface something genuinely non-obvious (a top-revenue customer or
    # product that's actually thin/negative on margin) instead of just
    # re-describing the charts.
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

    BUSINESS_PROMPT_TEMPLATE = """You are a sharp business advisor writing a short insights panel for a dashboard. The reader glances at this for a few seconds -- every insight has to earn its place by telling them something to DO, not something they could already see on the chart next to it.

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

    @st.cache_data(show_spinner=False)
    def generate_business_insights(facts_json: str):
        prompt = BUSINESS_PROMPT_TEMPLATE.format(facts=facts_json)
        return parse_insights(call_llm_json(prompt, temperature=0.5, max_tokens=700))

    try:
        with st.spinner("Generating insights..."):
            ai_insights = generate_business_insights(facts_json)
        cols = st.columns(len(ai_insights))
        for i, (col, insight) in enumerate(zip(cols, ai_insights)):
            render_insight_card(col, insight, "Business Insight", i)
    except Exception as e:
        st.error(f"Couldn't generate AI insights right now: {e}")


# =============================================================================
# Stock Market Dashboard (tab 2)
# =============================================================================

# Fixed name -> ticker mapping for the dropdown. A dict (not a bare list of
# symbols) so the picker can show a human-readable label while the rest of the
# code works with the plain ticker string yfinance expects.
STOCK_TICKERS = {
    "Apple (AAPL)": "AAPL",
    "Nike (NKE)": "NKE",
    "Tesla (TSLA)": "TSLA",
    "Disney (DIS)": "DIS",
    "McDonald's (MCD)": "MCD",
    "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN",
    "Coca-Cola (KO)": "KO",
    "Netflix (NFLX)": "NFLX",
    "Starbucks (SBUX)": "SBUX",
}

# Display-window options for the price chart/KPIs. The VALUE (days) is used to
# slice the cached 2-year fetch below -- see load_stock_history for why we
# always fetch more than this.
STOCK_PERIODS = {"1 Month": 30, "3 Months": 91, "6 Months": 182, "1 Year": 365, "2 Years": 730}


@st.cache_data(ttl=900, show_spinner="Fetching live price data...")
def load_stock_history(ticker: str) -> pd.DataFrame:
    """Fetch 2 years of daily price history for one ticker from yfinance.

    Always pulling 2 years (not just whatever the user has the period selector
    set to) means the 50-day moving average has a full 50 days of real lookback
    to compute from even when the display window is later narrowed to, say,
    "1 Month" -- otherwise the MA would be wrong or entirely NaN for a short
    window. ttl=900 caches each ticker's data for 15 minutes: still "live" (it
    refreshes on its own), but switching the period selector or re-running the
    script doesn't re-hit Yahoo Finance every time -- only a first load per
    ticker, or the first load after the cache expires, actually fetches.
    """
    hist = yf.Ticker(ticker).history(period="2y")
    if hist.empty:
        return hist
    hist = hist.reset_index()
    # yfinance returns tz-aware timestamps (exchange-local); we only need the
    # calendar date for slicing/plotting, so drop the tz to keep comparisons
    # against plain dates simple.
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
    hist["MA20"] = hist["Close"].rolling(window=20).mean()
    hist["MA50"] = hist["Close"].rolling(window=50).mean()
    return hist


def slice_to_period(hist: pd.DataFrame, days: int) -> pd.DataFrame:
    cutoff = hist["Date"].max() - pd.Timedelta(days=days)
    return hist[hist["Date"] >= cutoff].reset_index(drop=True)


def compute_stock_kpis(hist_full: pd.DataFrame, hist_window: pd.DataFrame) -> dict:
    latest = hist_full.iloc[-1]
    prev = hist_full.iloc[-2] if len(hist_full) >= 2 else None
    daily_change_dollars = (latest["Close"] - prev["Close"]) if prev is not None else None
    daily_change_pct = (daily_change_dollars / prev["Close"] * 100) if prev is not None and prev["Close"] else None

    period_start_price = hist_window["Close"].iloc[0]
    period_change_dollars = latest["Close"] - period_start_price
    period_change_pct = (period_change_dollars / period_start_price * 100) if period_start_price else None

    daily_returns = hist_window["Close"].pct_change().dropna()
    # Annualized volatility: the standard finance convention for expressing
    # day-to-day price swings as one comparable number (std of daily % returns,
    # scaled by sqrt(252) trading days/year). The "What am I looking at?"
    # expander below explains this in plain language.
    volatility_annualized = (daily_returns.std() * (252**0.5) * 100) if len(daily_returns) > 1 else None

    return {
        "price": latest["Close"],
        "daily_change_dollars": daily_change_dollars,
        "daily_change_pct": daily_change_pct,
        "period_change_dollars": period_change_dollars,
        "period_change_pct": period_change_pct,
        "volatility_annualized": volatility_annualized,
        "ma20": latest["MA20"],
        "ma50": latest["MA50"],
    }


def describe_ma_trend(price, ma20, ma50) -> str:
    # Pre-computed in Python (not left for the LLM to eyeball) so the AI
    # insights prompt gets a factual, already-correct comparison instead of
    # having to do arithmetic itself.
    if pd.isna(ma20) or pd.isna(ma50):
        return "Not enough price history yet to compare moving averages."
    if price > ma20 > ma50:
        return "Price is above both its 20-day and 50-day moving averages, and the 20-day average is above the 50-day average"
    if price < ma20 < ma50:
        return "Price is below both its 20-day and 50-day moving averages, and the 20-day average is below the 50-day average"
    if ma20 > ma50:
        return "The 20-day moving average is above the 50-day moving average, though price is not confirming both at once"
    return "The 20-day and 50-day moving averages are close together with no clear separation"


def render_price_chart(hist_window: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_window["Date"], y=hist_window["Close"], name="Price", mode="lines", line=dict(color=chart_palette[0], width=2)))
    fig.add_trace(go.Scatter(x=hist_window["Date"], y=hist_window["MA20"], name="20-Day MA", mode="lines", line=dict(color=chart_palette[1], width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=hist_window["Date"], y=hist_window["MA50"], name="50-Day MA", mode="lines", line=dict(color=chart_palette[2], width=2, dash="dash")))
    fig.update_layout(title=title)
    return style_trend_chart(fig, show_legend=True, x_title="Date", y_title="Price", y_tickformat=",.2f")


STOCK_PROMPT_TEMPLATE = """You are a patient finance tutor explaining a stock chart to a student, in plain English. You are NOT a financial advisor: never tell the reader to buy, sell, or hold, and never say something is a "good" or "bad" time to invest -- only explain what the data shows.

Using ONLY the facts given below, write 2 to 3 short insight sentences as JSON.

Formatting rule (critical): every number you mention MUST be copied character-for-character from the "value" fields below (same $, same %, same decimal places). Never round, abbreviate, or recompute a number yourself.

Output shape: {{"insights": [{{"headline": "...", "detail": "..."}}, ...]}}
- "headline": 2-5 words summarizing the insight (e.g. "Short-Term Uptrend", "High Volatility").
- "detail": ONE plain-English sentence, 15-30 words, explaining what the numbers mean -- trend direction, volatility level, or how the moving averages compare -- in a way a beginner investor would understand. Briefly unpack any jargon you use.
- Never mix a daily % change and a period % change in the same sentence without clearly labeling which is which (e.g. "up 3.5% today and 34.4% over the past 6 months") -- never state two bare percentages back to back.
- Always name which moving average you mean ("20-day" or "50-day") whenever you cite one -- never say just "its moving averages" without saying which.
- For volatility, give the reader a sense of scale using this general knowledge (background context, not one of the given facts): under ~20% annualized is relatively calm for a single stock, ~20-35% is moderate, above ~35% is high. Use it to describe the given volatility number as calm, moderate, or high -- do not invent a specific comparison number.
- Never give investment advice or a recommendation of any kind (no "buy", "sell", "should invest", "good/bad time to"); describe what is happening in the data, not what to do about it.
- If two tickers are present in the facts, at least one insight must directly compare them.
- Ground everything only in the numbers given -- never invent a cause not in the data.
- Plain text only: no markdown, no backticks, no bold/italic asterisks, no code formatting.

Facts:
{facts}
"""


@st.cache_data(show_spinner=False)
def generate_stock_insights(facts_json: str):
    prompt = STOCK_PROMPT_TEMPLATE.format(facts=facts_json)
    return parse_insights(call_llm_json(prompt, temperature=0.4, max_tokens=600))


def render_stock_ai_insights(facts: list):
    st.subheader("AI-Generated Insights")
    facts_json = json.dumps(facts, sort_keys=True)
    try:
        with st.spinner("Generating insights..."):
            ai_insights = generate_stock_insights(facts_json)
        cols = st.columns(len(ai_insights))
        for i, (col, insight) in enumerate(zip(cols, ai_insights)):
            render_insight_card(col, insight, "Market Insight", i)
    except Exception as e:
        st.error(f"Couldn't generate AI insights right now: {e}")


def render_stock_dashboard():
    st.header("Stock Market Dashboard")

    top_row_left, top_row_right = st.columns([2, 1])
    with top_row_left:
        view_mode = st.radio("View", [":material/show_chart: Single Stock", ":material/compare_arrows: Compare Two Stocks"], horizontal=True, label_visibility="collapsed")
    with top_row_right:
        if st.button(":material/refresh: Refresh Live Data", use_container_width=True):
            load_stock_history.clear()
            st.rerun()

    period_label = st.selectbox("Time Period", list(STOCK_PERIODS.keys()), index=2)
    period_days = STOCK_PERIODS[period_label]

    with st.expander(":material/help: What am I looking at?"):
        st.markdown(
            """
- **Ticker** -- the short letter code a company trades under on the stock market, e.g. `AAPL` for Apple or `TSLA` for Tesla. It's just an ID, like a username for a company's stock.
- **Moving Average (MA)** -- the average closing price over the last N trading days, recalculated every day. A "20-day MA" smooths out daily noise so you can see the short-term trend; a "50-day MA" shows the longer-term trend. When the price is above both, it's generally trending up; below both, trending down.
- **Volatility** -- how much the price swings day to day, shown here as an *annualized* percentage (a standard way to make different stocks/timeframes comparable). Higher volatility means bigger, more unpredictable price moves; lower volatility means calmer, steadier price action.
- **% Change** -- how much the price has moved, shown as a percentage instead of a raw dollar amount so it's comparable across stocks at very different price levels. "Daily % Change" is vs. yesterday's close; "Period % Change" is vs. the start of whatever time period you've selected above.
"""
        )

    if view_mode.endswith("Single Stock"):
        render_single_stock_view(period_label, period_days)
    else:
        render_compare_stocks_view(period_label, period_days)


def render_single_stock_view(period_label, period_days):
    ticker_name = st.selectbox("Company", list(STOCK_TICKERS.keys()), index=0)
    ticker = STOCK_TICKERS[ticker_name]

    hist_full = load_stock_history(ticker)
    if hist_full.empty:
        st.error(f"Couldn't load live data for {ticker} right now -- try again in a moment.")
        return
    hist_window = slice_to_period(hist_full, period_days)
    kpis = compute_stock_kpis(hist_full, hist_window)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        ":material/payments: Price",
        fmt_price(kpis["price"]),
        delta=fmt_change_pct(kpis["daily_change_pct"]),
        delta_description="vs previous close",
        border=True,
    )
    col2.metric(
        ":material/trending_up: Daily Change",
        fmt_price_change(kpis["daily_change_dollars"]) if kpis["daily_change_dollars"] is not None else "N/A",
        delta=fmt_change_pct(kpis["daily_change_pct"]),
        delta_description="vs previous close",
        border=True,
    )
    col3.metric(
        ":material/query_stats: Period Change",
        fmt_price_change(kpis["period_change_dollars"]),
        delta=fmt_change_pct(kpis["period_change_pct"]),
        delta_description=f"over {period_label}",
        border=True,
    )
    col4.metric(":material/candlestick_chart: Volatility (Annualized)", fmt_pct(kpis["volatility_annualized"]), border=True)

    st.plotly_chart(render_price_chart(hist_window, f"{ticker_name} Price with Moving Averages"), use_container_width=True)

    trend_text = describe_ma_trend(kpis["price"], kpis["ma20"], kpis["ma50"])
    facts = [
        {"label": "Ticker", "value": ticker_name},
        {"label": "Current Price", "value": fmt_price(kpis["price"])},
        {"label": "Daily % Change", "value": fmt_change_pct(kpis["daily_change_pct"])},
        {"label": f"% Change over {period_label}", "value": fmt_change_pct(kpis["period_change_pct"])},
        {"label": "Annualized Volatility", "value": fmt_pct(kpis["volatility_annualized"])},
        {"label": "20-Day Moving Average", "value": fmt_price(kpis["ma20"]) if not pd.isna(kpis["ma20"]) else "not enough history"},
        {"label": "50-Day Moving Average", "value": fmt_price(kpis["ma50"]) if not pd.isna(kpis["ma50"]) else "not enough history"},
        {"label": "How Price Compares to Its Moving Averages", "value": trend_text},
    ]
    render_stock_ai_insights(facts)


def render_compare_stocks_view(period_label, period_days):
    ticker_names = list(STOCK_TICKERS.keys())
    col_a, col_b = st.columns(2)
    name_a = col_a.selectbox("First Company", ticker_names, index=0)
    # Excludes whatever's picked in the first dropdown, so the two selections
    # can never collide -- no need to handle/display a "pick two different
    # companies" error state.
    remaining = [n for n in ticker_names if n != name_a]
    name_b = col_b.selectbox("Second Company", remaining, index=0)
    ticker_a, ticker_b = STOCK_TICKERS[name_a], STOCK_TICKERS[name_b]

    hist_a_full = load_stock_history(ticker_a)
    hist_b_full = load_stock_history(ticker_b)
    if hist_a_full.empty or hist_b_full.empty:
        st.error("Couldn't load live data for one of these tickers right now -- try again in a moment.")
        return

    hist_a = slice_to_period(hist_a_full, period_days)
    hist_b = slice_to_period(hist_b_full, period_days)
    kpis_a = compute_stock_kpis(hist_a_full, hist_a)
    kpis_b = compute_stock_kpis(hist_b_full, hist_b)

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    kpi_col1.metric(f":material/payments: {ticker_a} Price", fmt_price(kpis_a["price"]), delta=fmt_change_pct(kpis_a["period_change_pct"]), delta_description=f"over {period_label}", border=True)
    kpi_col2.metric(f":material/payments: {ticker_b} Price", fmt_price(kpis_b["price"]), delta=fmt_change_pct(kpis_b["period_change_pct"]), delta_description=f"over {period_label}", border=True)
    kpi_col3.metric(f":material/trending_up: {ticker_a} Daily Change", fmt_change_pct(kpis_a["daily_change_pct"]), delta=fmt_change_pct(kpis_a["daily_change_pct"]), delta_description="vs previous close", border=True)
    kpi_col4.metric(f":material/trending_up: {ticker_b} Daily Change", fmt_change_pct(kpis_b["daily_change_pct"]), delta=fmt_change_pct(kpis_b["daily_change_pct"]), delta_description="vs previous close", border=True)

    # Two stocks at different price levels can't be overlaid as raw dollar
    # lines and mean anything -- a $300 stock moving $3 looks identical to a
    # $30 stock moving $3 even though one just moved 10x more. Rebasing both
    # to "% change since the start of the selected window" puts them on one
    # shared, actually-comparable y-axis.
    rebased_a = hist_a.copy()
    rebased_a["Pct Change"] = (rebased_a["Close"] / rebased_a["Close"].iloc[0] - 1) * 100
    rebased_b = hist_b.copy()
    rebased_b["Pct Change"] = (rebased_b["Close"] / rebased_b["Close"].iloc[0] - 1) * 100

    fig_compare = go.Figure()
    fig_compare.add_trace(go.Scatter(x=rebased_a["Date"], y=rebased_a["Pct Change"], name=ticker_a, mode="lines", line=dict(color=chart_palette[0], width=2)))
    fig_compare.add_trace(go.Scatter(x=rebased_b["Date"], y=rebased_b["Pct Change"], name=ticker_b, mode="lines", line=dict(color=chart_palette[1], width=2)))
    fig_compare.update_layout(title=f"{ticker_a} vs {ticker_b}: % Change Over {period_label}")
    fig_compare = style_trend_chart(fig_compare, show_legend=True, x_title="Date", y_title="% Change Since Start", y_tickformat="+.1f", y_tickprefix="", y_ticksuffix="%", y_zeroline=True)
    st.plotly_chart(fig_compare, use_container_width=True)

    trend_a = describe_ma_trend(kpis_a["price"], kpis_a["ma20"], kpis_a["ma50"])
    trend_b = describe_ma_trend(kpis_b["price"], kpis_b["ma20"], kpis_b["ma50"])
    facts = [
        {"label": f"{ticker_a} % Change over {period_label}", "value": fmt_change_pct(kpis_a["period_change_pct"])},
        {"label": f"{ticker_b} % Change over {period_label}", "value": fmt_change_pct(kpis_b["period_change_pct"])},
        {"label": f"{ticker_a} Annualized Volatility", "value": fmt_pct(kpis_a["volatility_annualized"])},
        {"label": f"{ticker_b} Annualized Volatility", "value": fmt_pct(kpis_b["volatility_annualized"])},
        {"label": f"{ticker_a} vs Its Moving Averages", "value": trend_a},
        {"label": f"{ticker_b} vs Its Moving Averages", "value": trend_b},
    ]
    render_stock_ai_insights(facts)


# =============================================================================
# Churn Radar (tab 3)
# =============================================================================

# Superstore has no explicit churn column, so one is derived from recency: a
# customer is "churned" if their most recent order is more than this many
# days before the last order date anywhere in the dataset. 180 days (~6
# months) is a common real-world "gone quiet" cutoff for a retail business,
# and empirically lands this dataset at a ~25% churn rate -- enough churned
# examples for a classifier to learn from, without being so rare (or so
# common) that the label is nearly useless.
CHURN_RECENCY_THRESHOLD_DAYS = 180
CHURN_FEATURE_COLS = ["Total Orders", "Total Sales", "Total Profit", "Avg Discount", "Total Quantity", "Tenure Days", "Avg Order Value"]


@st.cache_data
def build_churn_dataset() -> pd.DataFrame:
    """Roll the order-line-level Superstore data up to one row per customer,
    with RFM-style features (recency/frequency/monetary) and a derived churn
    label. Recency itself is deliberately EXCLUDED from CHURN_FEATURE_COLS
    below -- it's what defines the label, so training on it would be circular
    (the model would just learn "Recency > 180 -> churned", which isn't a
    real prediction, it's reading the label off itself).
    """
    df = load_data()
    max_date = df["Order Date"].max()

    customers = df.groupby("Customer ID", as_index=False).agg(**{
        "Customer Name": ("Customer Name", "first"),
        "Last Order Date": ("Order Date", "max"),
        "First Order Date": ("Order Date", "min"),
        "Total Orders": ("Order ID", "nunique"),
        "Total Sales": ("Sales", "sum"),
        "Total Profit": ("Profit", "sum"),
        "Avg Discount": ("Discount", "mean"),
        "Total Quantity": ("Quantity", "sum"),
    })
    customers["Recency Days"] = (max_date - customers["Last Order Date"]).dt.days
    customers["Tenure Days"] = (max_date - customers["First Order Date"]).dt.days
    customers["Avg Order Value"] = customers["Total Sales"] / customers["Total Orders"]
    customers["Churned"] = (customers["Recency Days"] > CHURN_RECENCY_THRESHOLD_DAYS).astype(int)
    return customers


@st.cache_data(show_spinner="Training churn models...")
def train_churn_models(customers: pd.DataFrame) -> dict:
    """Train a Decision Tree and a Random Forest on the same train/test split
    so their accuracy is directly comparable, then score every customer (not
    just the held-out test set) with the Random Forest for the "Customers to
    Call First" table -- the test split is for honestly measuring accuracy,
    but the actual retention list should cover the whole customer base.
    Cached on the `customers` DataFrame: same input data + the fixed
    random_state below always produces the same models, so this only
    actually retrains once.
    """
    X = customers[CHURN_FEATURE_COLS]
    y = customers["Churned"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    dt = DecisionTreeClassifier(max_depth=6, random_state=42)
    dt.fit(X_train, y_train)
    dt_accuracy = accuracy_score(y_test, dt.predict(X_test))

    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)

    # Honest baseline: the accuracy you'd get by always guessing the majority
    # class ("not churned") without looking at any features at all. A model
    # that doesn't clear this bar isn't actually learning anything useful.
    baseline_accuracy = max(y_test.mean(), 1 - y_test.mean())

    return {
        "dt_accuracy": dt_accuracy,
        "rf_accuracy": rf_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "confusion_matrix": confusion_matrix(y_test, rf_pred, labels=[0, 1]),
        "feature_importances": pd.Series(rf.feature_importances_, index=CHURN_FEATURE_COLS).sort_values(ascending=False),
        "all_probabilities": rf.predict_proba(X)[:, 1],
    }


def render_churn_radar_dashboard():
    st.header("Churn Radar")
    st.caption(
        f"Superstore has no built-in churn label, so one is derived here: a customer is marked "
        f"**churned** if their most recent order was more than {CHURN_RECENCY_THRESHOLD_DAYS} days "
        f"before the last order date in the whole dataset."
    )

    customers = build_churn_dataset()
    results = train_churn_models(customers)

    st.subheader("Model Comparison")
    model_col1, model_col2 = st.columns(2)
    model_col1.metric(":material/account_tree: Decision Tree Accuracy", fmt_pct(results["dt_accuracy"] * 100), border=True)
    model_col2.metric(":material/forest: Random Forest Accuracy", fmt_pct(results["rf_accuracy"] * 100), border=True)
    st.caption(
        f"For context: always guessing \"not churned\" for every customer would score "
        f"{fmt_pct(results['baseline_accuracy'] * 100)} accuracy on this same test set -- a model "
        f"needs to clear that bar to be worth using, not just post a big-sounding number."
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Confusion Matrix (Random Forest)**")
        cm = results["confusion_matrix"]
        # A heatmap doesn't fit style_bar_chart/style_trend_chart's shape (both
        # assume one value axis against one category/time axis), so it's
        # hand-styled here using the same tokens (palette, ink, transparent
        # background) rather than raw Plotly defaults.
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Predicted Active", "Predicted Churned"],
                y=["Actual Active", "Actual Churned"],
                colorscale=[[0, "rgba(0,0,0,0)"], [1, chart_palette[0]]],
                text=cm,
                texttemplate="%{text}",
                textfont=dict(size=22, color=secondary_ink),
                showscale=False,
                xgap=3,
                ygap=3,
            )
        )
        fig_cm.update_layout(
            font=dict(size=13, color=secondary_ink),
            xaxis=dict(tickfont=dict(size=13, color=muted_ink), showgrid=False),
            yaxis=dict(tickfont=dict(size=13, color=muted_ink), showgrid=False, autorange="reversed"),
            margin=dict(t=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("Rows are what actually happened, columns are what the model predicted.")

    with chart_col2:
        importance_df = results["feature_importances"].reset_index()
        importance_df.columns = ["Feature", "Importance"]
        # An explicit, non-empty title (matching the pattern the other bar
        # charts in this file already use) instead of a separate st.markdown
        # header -- a Plotly figure with no title text renders a stray
        # "undefined" in its place once style_bar_chart's update_layout call
        # touches the title's font.
        fig_importance = px.bar(importance_df.head(10), x="Feature", y="Importance", title="Top Churn Drivers")
        st.plotly_chart(style_bar_chart(fig_importance, "Importance", y_tickprefix="", y_tickformat=".0%"), use_container_width=True)

    st.subheader("Customers to Call First")
    st.caption("Currently-active customers ranked by predicted churn risk -- an early-warning list, not customers who've already gone quiet.")
    scored = customers.copy()
    scored["Churn Probability"] = results["all_probabilities"] * 100
    to_call = scored[scored["Churned"] == 0].sort_values("Churn Probability", ascending=False).head(15)[
        ["Customer Name", "Recency Days", "Total Orders", "Total Sales", "Total Profit", "Churn Probability"]
    ].reset_index(drop=True)
    to_call.index += 1
    to_call.index.name = "Rank"

    st.dataframe(
        to_call,
        column_config={
            "Total Sales": st.column_config.NumberColumn("Total Sales", format="$%.2f"),
            "Total Profit": st.column_config.NumberColumn("Total Profit", format="$%.2f"),
            "Churn Probability": st.column_config.NumberColumn("Churn Probability", format="%.1f%%"),
        },
        use_container_width=True,
    )


# =============================================================================
# Sales Forecast (tab 4)
# =============================================================================

FORECAST_LAGS = [1, 2, 3]
FORECAST_TEST_MONTHS = 6


@st.cache_data(show_spinner="Fitting forecast model...")
def build_sales_forecast() -> dict:
    """Build lag features from the monthly sales series and fit a plain linear
    regression against a naive "same as last month" baseline. The split is
    chronological (last FORECAST_TEST_MONTHS months held out), not a random
    train_test_split -- shuffling months would let the model "forecast" a
    month using data that (in a real forecast) wouldn't exist yet.
    """
    df = load_data()
    monthly = df.groupby("Order Month", as_index=False)["Sales"].sum().sort_values("Order Month").reset_index(drop=True)

    lagged = monthly.copy()
    for lag in FORECAST_LAGS:
        lagged[f"Lag {lag}"] = lagged["Sales"].shift(lag)
    lagged = lagged.dropna().reset_index(drop=True)

    lag_cols = [f"Lag {lag}" for lag in FORECAST_LAGS]
    train = lagged.iloc[:-FORECAST_TEST_MONTHS]
    test = lagged.iloc[-FORECAST_TEST_MONTHS:].copy()

    model = LinearRegression()
    model.fit(train[lag_cols], train["Sales"])
    test["Forecast"] = model.predict(test[lag_cols])
    test["Naive Baseline"] = test["Lag 1"]  # "next month = same as last month"

    return {
        "monthly": monthly,
        "test": test,
        "model_mae": mean_absolute_error(test["Sales"], test["Forecast"]),
        "naive_mae": mean_absolute_error(test["Sales"], test["Naive Baseline"]),
        "months_used": len(lagged),
    }


def render_sales_forecast_dashboard():
    st.header("Sales Forecast")
    st.caption(f"A simple linear regression on the last {len(FORECAST_LAGS)} months of sales, evaluated against a naive \"same as last month\" baseline over the most recent {FORECAST_TEST_MONTHS} months.")

    results = build_sales_forecast()
    test = results["test"]
    model_wins = results["model_mae"] < results["naive_mae"]

    col1, col2 = st.columns(2)
    col1.metric(":material/model_training: Model MAE", fmt_money(results["model_mae"]), border=True)
    col2.metric(":material/rule: Naive Baseline MAE", fmt_money(results["naive_mae"]), border=True)

    # MAE (mean absolute error) is in dollars, same units as sales -- lower is
    # better. Reported honestly either way: this dataset has a small number of
    # months and strong holiday seasonality a 3-month-lag linear model doesn't
    # capture well, so the naive baseline sometimes wins, and that result is
    # shown as-is rather than only reported when the model happens to win.
    # strip_markdown escapes "$" -- any two dollar-prefixed figures in one
    # string are otherwise treated as a pair of LaTeX math delimiters by
    # Streamlit's markdown renderer, silently eating both "$" signs and
    # rendering the text between them as an equation (the same bug fixed
    # earlier in the AI insight cards, but it applies to any st.markdown-family
    # text, hand-written or LLM-generated).
    if model_wins:
        st.success(strip_markdown(f"The model beat the naive baseline on this test window ({fmt_money(results['model_mae'])} vs {fmt_money(results['naive_mae'])} average error)."))
    else:
        st.warning(strip_markdown(f"The naive baseline actually beat the model on this test window ({fmt_money(results['naive_mae'])} vs {fmt_money(results['model_mae'])} average error) -- see the caveats below for why."))

    monthly = results["monthly"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["Order Month"], y=monthly["Sales"], name="Actual", mode="lines", line=dict(color=chart_palette[0], width=2)))
    fig.add_trace(go.Scatter(x=test["Order Month"], y=test["Forecast"], name="Model Forecast", mode="lines+markers", line=dict(color=chart_palette[1], width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=test["Order Month"], y=test["Naive Baseline"], name="Naive Baseline", mode="lines+markers", line=dict(color=chart_palette[2], width=2, dash="dot")))
    fig.update_layout(title="Monthly Sales: Actual vs Forecast")
    st.plotly_chart(style_trend_chart(fig, show_legend=True, x_title="Month", y_title="Sales"), use_container_width=True)

    st.subheader("Caveats")
    st.info(
        f"""
- **Small data size** -- this forecast is fit on only {results['months_used']} months of history, with just {FORECAST_TEST_MONTHS} months held out to test on. That's not enough data for the model to reliably separate a real pattern from noise.
- **Seasonality** -- sales spike sharply around the holidays every year. A model that only looks at the last {len(FORECAST_LAGS)} months has no way to know "December is always big" -- it only sees recent numbers, not the time of year.
- **Unforeseen events** -- no model trained on past sales can see a supply shortage, a new competitor, a marketing campaign, or a economic shift coming. Every forecast here is "if the future looks like the recent past," which is never guaranteed.
"""
    )


# =============================================================================
# The Prediction Experiment (tab 5)
# =============================================================================

PREDICTION_FEATURE_COLS = ["Daily Return", "Return MA5", "Return MA10", "MA20 Above MA50"]
PREDICTION_TEST_FRACTION = 0.2


@st.cache_data(show_spinner="Training prediction model...")
def build_prediction_experiment(ticker: str) -> dict:
    """Engineer next-day-direction features from one ticker's cached price
    history (load_stock_history is already @st.cache_data'd, so this only
    re-touches yfinance if that cache has expired) and evaluate a Random
    Forest against two baselines on a chronological holdout. The split is by
    date, not train_test_split's random shuffle -- shuffling days would let
    the model "predict" a day using information from days that come after it,
    which could never happen in a real forecast.
    """
    hist = load_stock_history(ticker).copy()
    hist["Daily Return"] = hist["Close"].pct_change()
    hist["Return MA5"] = hist["Daily Return"].rolling(5).mean()
    hist["Return MA10"] = hist["Daily Return"].rolling(10).mean()
    hist["MA20 Above MA50"] = (hist["MA20"] > hist["MA50"]).astype(int)
    # Tomorrow's direction, known only in hindsight -- this is the target,
    # never a feature the model gets to see.
    hist["Target"] = (hist["Close"].shift(-1) > hist["Close"]).astype(int)
    hist = hist.dropna().reset_index(drop=True)

    test_size = max(int(len(hist) * PREDICTION_TEST_FRACTION), 10)
    train = hist.iloc[:-test_size]
    test = hist.iloc[-test_size:]

    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(train[PREDICTION_FEATURE_COLS], train["Target"])
    predictions = model.predict(test[PREDICTION_FEATURE_COLS])

    model_accuracy = accuracy_score(test["Target"], predictions)
    pct_days_up = test["Target"].mean()
    # "Always predict up" only looks smart when most test days actually were
    # up (or, symmetrically, down) -- its real accuracy is whichever class was
    # more common in that specific window, not a fixed number.
    always_up_accuracy = max(pct_days_up, 1 - pct_days_up)

    # labels=[0, 1] fixes the layout to [[TN, FP], [FN, TP]] (row = actual
    # Down/Up, column = predicted Down/Up) -- unpacked here once so both the
    # heatmap and the AI-insights facts list read named counts, not raw
    # matrix indices.
    cm = confusion_matrix(test["Target"], predictions, labels=[0, 1])
    true_negatives, false_positives = cm[0]
    false_negatives, true_positives = cm[1]

    return {
        "model_accuracy": model_accuracy,
        "coin_flip_accuracy": 0.5,
        "always_up_accuracy": always_up_accuracy,
        "pct_days_up": pct_days_up,
        "confusion_matrix": cm,
        "true_positives": int(true_positives),
        "true_negatives": int(true_negatives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "test_days": len(test),
        "train_days": len(train),
    }


PREDICTION_PROMPT_TEMPLATE = """You are a patient finance tutor explaining the result of a prediction experiment to a student, in plain English. You are NOT a financial advisor: never claim a real trading edge exists, never say something is a "good" or "bad" time to invest, and never suggest a trade or strategy.

Using ONLY the facts given below, write exactly 3 insights as JSON.

Formatting rule (critical): every number you mention MUST be copied character-for-character from the "value" fields below (same %, same counts, same decimal places). Never round, abbreviate, or recompute a number yourself.

Output shape: {{"insights": [{{"headline": "...", "detail": "..."}}, ...]}}
- "headline": 2-5 words summarizing the insight.
- "detail": ONE plain-English sentence, 15-35 words.
- One insight must state plainly whether the model beat both baselines, one of them, or neither -- using only the "Did the Model Beat..." facts and the accuracy numbers given, never recomputed or guessed at.
- One insight must explain, in simple age-appropriate terms, WHY beating these baselines is hard: if a pattern like this reliably worked, people would already be trading on it until that trading itself made the pattern disappear (the "efficient markets" idea) -- explain the idea plainly, don't just name-drop the term.
- One insight must cover what this result does and doesn't prove: NEVER claim a real, repeatable trading edge was found even if the model beat both baselines here -- one company and a small test window is not proof of anything repeatable. If the model did NOT clear the baselines, frame that plainly as a legitimate, useful finding about how hard this problem is, not as a failure.
- Never suggest a trade, a strategy, or that this result predicts future performance.
- Ground everything only in the numbers given -- never invent a cause not in the data.
- Plain text only: no markdown, no backticks, no bold/italic asterisks, no code formatting.

Facts:
{facts}
"""


@st.cache_data(show_spinner=False)
def generate_prediction_insights(facts_json: str):
    prompt = PREDICTION_PROMPT_TEMPLATE.format(facts=facts_json)
    return parse_insights(call_llm_json(prompt, temperature=0.4, max_tokens=600))


def render_prediction_experiment_dashboard():
    st.header("The Prediction Experiment")
    st.caption(
        "An honest test: can a Random Forest predict whether a stock closes UP or DOWN tomorrow, "
        "using only that stock's own recent price behavior? Trained and evaluated on one ticker at a time."
    )
    st.info(":material/school: **Educational simulation -- not investment advice.** This tests a modeling idea; it does not recommend any trade.")

    ticker_name = st.selectbox("Company", list(STOCK_TICKERS.keys()), index=0, key="prediction_ticker")
    ticker = STOCK_TICKERS[ticker_name]

    hist_check = load_stock_history(ticker)
    if hist_check.empty:
        st.error(f"Couldn't load live data for {ticker} right now -- try again in a moment.")
        return

    results = build_prediction_experiment(ticker)

    st.subheader("Accuracy vs. Baselines")
    col1, col2, col3 = st.columns(3)
    col1.metric(":material/model_training: Model Accuracy", fmt_pct(results["model_accuracy"] * 100), border=True)
    col2.metric(":material/casino: Coin Flip", fmt_pct(results["coin_flip_accuracy"] * 100), border=True)
    col3.metric(":material/trending_up: Always Predict Up", fmt_pct(results["always_up_accuracy"] * 100), border=True)
    st.caption(
        f"\"Always predict up\" scores {fmt_pct(results['always_up_accuracy'] * 100)} here because "
        f"{fmt_pct(results['pct_days_up'] * 100)} of the {results['test_days']} test days were actually up days -- "
        f"not because it's smart, just because one direction happened to be more common in this window."
    )

    st.markdown("**Confusion Matrix**")
    cm = results["confusion_matrix"]
    # Same hand-styled heatmap approach as Churn Radar's confusion matrix --
    # a heatmap doesn't fit style_bar_chart/style_trend_chart's shape (both
    # assume one value axis against one category/time axis), so this reuses
    # the same tokens (palette, ink, transparent background) directly instead.
    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Predicted Down", "Predicted Up"],
            y=["Actual Down", "Actual Up"],
            colorscale=[[0, "rgba(0,0,0,0)"], [1, chart_palette[0]]],
            text=cm,
            texttemplate="%{text}",
            textfont=dict(size=22, color=secondary_ink),
            showscale=False,
            xgap=3,
            ygap=3,
        )
    )
    fig_cm.update_layout(
        font=dict(size=13, color=secondary_ink),
        xaxis=dict(tickfont=dict(size=13, color=muted_ink), showgrid=False),
        yaxis=dict(tickfont=dict(size=13, color=muted_ink), showgrid=False, autorange="reversed"),
        margin=dict(t=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_cm, use_container_width=True)
    st.caption("Rows are what actually happened, columns are what the model predicted.")

    st.subheader("What This Does and Doesn't Show")

    model_beat_coin_flip = results["model_accuracy"] > results["coin_flip_accuracy"]
    model_beat_always_up = results["model_accuracy"] > results["always_up_accuracy"]

    # Same "pre-computed facts, not raw numbers" approach as the Business and
    # Stock sections: the model already knows whether it beat each baseline
    # (Python did the comparison), so the LLM only has to explain that
    # outcome in plain English, never recompute or guess at it.
    facts = [
        {"label": "Company", "value": ticker_name},
        {"label": "Test Window Size", "value": f"{results['test_days']} trading days"},
        {"label": "Model Accuracy", "value": fmt_pct(results["model_accuracy"] * 100)},
        {"label": "Coin Flip Baseline", "value": fmt_pct(results["coin_flip_accuracy"] * 100)},
        {"label": "Always-Predict-Up Baseline", "value": fmt_pct(results["always_up_accuracy"] * 100)},
        {"label": "Did the Model Beat the Coin Flip Baseline", "value": "Yes" if model_beat_coin_flip else "No"},
        {"label": "Did the Model Beat the Always-Predict-Up Baseline", "value": "Yes" if model_beat_always_up else "No"},
        {"label": "True Positives (Predicted Up, Was Actually Up)", "value": str(results["true_positives"])},
        {"label": "True Negatives (Predicted Down, Was Actually Down)", "value": str(results["true_negatives"])},
        {"label": "False Positives (Predicted Up, Was Actually Down)", "value": str(results["false_positives"])},
        {"label": "False Negatives (Predicted Down, Was Actually Up)", "value": str(results["false_negatives"])},
    ]
    # Sort keys so the JSON string is stable for a given result -- this
    # string is the cache key below, so switching tickers and switching back
    # reuses the cached call instead of re-hitting the API.
    facts_json = json.dumps(facts, sort_keys=True)

    try:
        with st.spinner("Generating insights..."):
            ai_insights = generate_prediction_insights(facts_json)
        cols = st.columns(len(ai_insights))
        for i, (col, insight) in enumerate(zip(cols, ai_insights)):
            render_insight_card(col, insight, "Prediction Insight", i)
    except Exception as e:
        st.error(f"Couldn't generate AI insights right now: {e}")


# =============================================================================
# Tabs -- all dashboards live in this one app, one script, one file.
# =============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        ":material/store: Business Performance",
        ":material/candlestick_chart: Stock Market",
        ":material/radar: Churn Radar",
        ":material/timeline: Sales Forecast",
        ":material/psychology: Prediction Experiment",
    ]
)

with tab1:
    render_business_dashboard()

with tab2:
    render_stock_dashboard()

with tab3:
    render_churn_radar_dashboard()

with tab4:
    render_sales_forecast_dashboard()

with tab5:
    render_prediction_experiment_dashboard()
