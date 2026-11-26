# ---------- 1. IMPORTS ----------
import numpy as np                           # numbers and maths
import pandas as pd                          # tables (dataframes)
import plotly.graph_objects as go            # charts
import streamlit as st                       # the web app itself
from sklearn.linear_model import LinearRegression  # the price model

# ---------- 2. PAGE SETTINGS ----------
# Must be the first Streamlit command. Sets the browser tab title, icon and width.
st.set_page_config(page_title="Farmsharp", page_icon="🌽", layout="wide")

# ---------- 3. CONSTANTS ----------
DATA_PATH = "data/farmsharp.csv"             # where the Ilorin maize data lives
PRICE = "maize_fao"                          # the price column we predict (N/kg)
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# The inputs the model learns from (same as the notebook)
FEATURES = ["month_sin", "month_cos", "t", "p0", "lag1", "lag2",
            "roll3", "infl", "trust", "range"]
# Farm cost boxes on the ProfitWise page: (label, starting value in Naira)
COSTS = [("Seeds", 15000), ("Fertilizer", 40000), ("Labourers", 35000),
         ("Pesticides and herbicides", 0), ("Land preparation", 0),
         ("Transport", 0), ("Others", 12000)]


def naira(value):
    """Show a number like ₦437."""
    return f"₦{value:,.0f}"


def month_sin_cos(month):
    """Turn a month number (1-12) into a circle position so Dec sits next to Jan."""
    return np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)


# ---------- 4. LOAD DATA AND TRAIN THE MODEL ----------
@st.cache_data  # run once, then remember the result so the app stays fast
def load_and_train():
    raw = pd.read_csv(DATA_PATH)                                   # read the CSV
    raw = raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed")])  # drop index column
    d = raw.copy()
    d["date"] = pd.to_datetime(d["price_date"].astype(str), format="%Y%m%d")  # 20261001 -> a date
    d = d.sort_values("date").reset_index(drop=True)               # oldest first

    log_price = np.log(d[PRICE])                                   # work in logs so changes are in %
    next_month = d["date"].dt.month % 12 + 1                       # the month we are predicting
    d["month_sin"], d["month_cos"] = month_sin_cos(next_month)     # season features
    d["t"] = np.arange(len(d))                                     # time trend (0, 1, 2 ...)
    d["p0"] = log_price                                            # this month's price
    d["lag1"] = log_price.shift(1)                                 # last month's price
    d["lag2"] = log_price.shift(2)                                 # two months ago
    d["roll3"] = log_price.rolling(3).mean()                       # 3-month average
    d["infl"] = d["inflation_maize_fao"]                           # inflation
    d["trust"] = d["trust_maize_fao"]                              # data trust score
    d["range"] = (d["h_maize_fao"] - d["l_maize_fao"]) / d[PRICE]  # how spread out prices were
    d["target"] = log_price.shift(-1)                              # the answer: next month's price

    train = d.dropna(subset=FEATURES + ["target"])                 # keep rows with no gaps
    model = LinearRegression().fit(train[FEATURES], train["target"])  # learn
    return d, model


def price_range(prices, h, window=60):
    """Low and high multipliers from how much prices really moved h months ahead."""
    logs = np.log(np.asarray(prices))
    moves = (logs[h:] - logs[:-h])[-window:]                       # real h-month changes
    low, mid, high = np.percentile(moves, [5, 50, 95])             # 5th, middle, 95th
    return np.exp(low - mid), np.exp(high - mid)


def make_forecast(d, model, months=3):
    """Predict the next `months` months, one at a time."""
    last = d.iloc[-1]                                              # latest month in the data
    history = list(np.log(d[PRICE]))                               # prices so far (logs)
    rows = []
    for h in range(1, months + 1):
        date = last["date"] + pd.DateOffset(months=h)              # the month we predict
        s, c = month_sin_cos(date.month)
        x = pd.DataFrame([{
            "month_sin": s, "month_cos": c, "t": last["t"] + h - 1,
            "p0": history[-1], "lag1": history[-2], "lag2": history[-3],
            "roll3": np.mean(history[-3:]),
            "infl": last["infl"], "trust": last["trust"], "range": last["range"],
        }])
        pred = float(model.predict(x[FEATURES])[0])                # predicted log price
        history.append(pred)                                       # feed it into the next step
        low, high = price_range(d[PRICE], h)
        price = np.exp(pred)
        rows.append({"Month": date.strftime("%b %Y"), "Date": date,
                     "Expected": price, "Low": price * low, "High": price * high})
    return pd.DataFrame(rows)


def season_index(d):
    """Average price of each calendar month vs its yearly average (last 10 years)."""
    s = d.set_index("date")[PRICE]
    ratio = (s / s.rolling(12, center=True).mean()).dropna().tail(120)
    return ratio.groupby(ratio.index.month).mean()                 # index 1..12


def profitwise(total_cost, kg, margin, market, low_case, best_month):
    """Return (cost per kg, suggested price, level, message)."""
    cost_per_kg = total_cost / kg
    suggested = cost_per_kg * (1 + margin)
    if market < cost_per_kg:
        return cost_per_kg, suggested, "error", f"Market price is below your cost. Selling now is a loss. Best month to sell: {best_month}."
    if low_case < cost_per_kg:
        return cost_per_kg, suggested, "warning", f"Profitable now, but in a low case ({naira(low_case)}/kg) you would lose money. Consider selling early or in {best_month}."
    if suggested > market * 1.1:
        return cost_per_kg, suggested, "warning", "Your price is above the market. Buyers may leave. Lower your margin."
    if suggested < market * 0.9:
        return cost_per_kg, suggested, "success", f"You can charge more. The market pays about {naira(market)}/kg."
    return cost_per_kg, suggested, "success", "Good range. Your price is close to the market."


# ---------- 5. PREPARE EVERYTHING ----------
d, model = load_and_train()
fc = make_forecast(d, model, months=3)
season = season_index(d)
best_month = MONTHS[int(season.idxmax()) - 1]                      # highest seasonal price
cheap_month = MONTHS[int(season.idxmin()) - 1]                     # lowest seasonal price
now_price = float(d[PRICE].iloc[-1])

# ---------- 6. PAGE SWITCHING ----------
st.sidebar.image("assets/logo.png")                                # logo in the sidebar
page = st.sidebar.radio("Go to", ["Market and forecast", "ProfitWise"])  # the only way to switch pages
st.image("assets/logo.png", width=340)                                  # logo at the top of the page
st.caption("Maize price forecast for Ilorin, Kwara State")

# ---------- 7. PAGE 1: MARKET AND FORECAST ----------
if page == "Market and forecast":
    st.subheader("🌽 Market now")
    a, b, c = st.columns(3)
    a.metric("Per kg now", naira(now_price))
    b.metric("Expected next month", naira(fc["Expected"][0]))
    c.metric("Best month to sell", best_month, help=f"Cheapest month usually: {cheap_month}")

    st.subheader("Next 3 months")
    recent = d.tail(24)                                            # last 24 months of history
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent["date"], y=recent[PRICE], name="History",
                             line=dict(color="#4c8bf5", width=3)))
    fig.add_trace(go.Scatter(x=pd.concat([fc["Date"], fc["Date"][::-1]]),
                             y=pd.concat([fc["High"], fc["Low"][::-1]]),
                             fill="toself", mode="lines", name="Low to high range",
                             line=dict(width=0), fillcolor="rgba(201,150,26,0.25)"))
    fig.add_trace(go.Scatter(x=[recent["date"].iloc[-1]] + list(fc["Date"]),
                             y=[recent[PRICE].iloc[-1]] + list(fc["Expected"]),
                             name="Forecast", line=dict(color="#c9961a", width=3, dash="dash")))
    fig.update_layout(yaxis_title="Naira per kg", margin=dict(t=10), height=380)
    st.plotly_chart(fig, width="stretch")

    table = fc[["Month", "Expected", "Low", "High"]].copy()
    for col in ["Expected", "Low", "High"]:
        table[col] = table[col].map(naira)
    st.dataframe(table, hide_index=True, width="stretch")

    st.subheader("Best month to sell")
    st.caption("Average price in each month vs its yearly average, last 10 years. Above 1.0 is a good month to sell.")
    bar = go.Figure(go.Bar(x=MONTHS, y=season.values,
                           marker_color=["#c9961a" if m == best_month else "#9aa3b5" for m in MONTHS]))
    bar.add_hline(y=1.0, line_dash="dot")
    bar.update_layout(yaxis_range=[0.85, 1.1], margin=dict(t=10), height=300)
    st.plotly_chart(bar, width="stretch")

# ---------- 8. PAGE 2: PROFITWISE ----------
else:
    st.subheader("ProfitWise")
    st.write("Enter your costs. Farmsharp suggests a price and tells you if the market agrees.")
    cols = st.columns(3)
    total = 0
    for i, (label, default) in enumerate(COSTS):                   # one box per cost
        total += cols[i % 3].number_input(f"{label} (₦)", min_value=0, value=default, step=1000)
    kg = cols[0].number_input("Maize to sell (kg)", min_value=1, value=1000, step=50)
    margin = cols[1].number_input("Profit margin (%)", min_value=0, value=20, step=1) / 100

    market, low_case = float(fc["Expected"][0]), float(fc["Low"][0])
    cost_kg, suggested, level, message = profitwise(total, kg, margin, market, low_case, best_month)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Your cost per kg", naira(cost_kg))
    m2.metric("Suggested price per kg", naira(suggested))
    m3.metric("Market price per kg (next month)", naira(market))
    m4.metric("Low case per kg", naira(low_case))
    getattr(st, level)(message)                                    # st.success / st.warning / st.error
