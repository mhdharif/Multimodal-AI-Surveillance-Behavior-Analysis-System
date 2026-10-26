import streamlit as st
import pandas as pd
import os
from PIL import Image
# from rag.rag_engine import ask_rag
from query.query_engine import process_query

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Surveillance Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# ==========================================
# DARK THEME CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
    color: white;
}

div[data-testid="metric-container"] {
    background-color: #1A1D24;
    border: 1px solid #2E3440;
    padding: 15px;
    border-radius: 12px;
}

div[data-testid="stDataFrame"] {
    background-color: #1A1D24;
    border-radius: 12px;
    padding: 10px;
}

h1, h2, h3 {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================

st.markdown("""
<h1 style='text-align:center;'>
🛡️ AI Surveillance & Behavior Analysis System
</h1>

<hr>
""", unsafe_allow_html=True)

# ==========================================
# LOAD CSV
# ==========================================

log_file = "data/logs/alerts.csv"

if not os.path.exists(log_file):
    st.error("alerts.csv not found")
    st.stop()

try:
    df = pd.read_csv(log_file)
except pd.errors.EmptyDataError:
    st.warning("No incidents logged yet.")
    st.stop()

# ==========================================
# CHAT MEMORY
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# METRICS
# ==========================================

high_risk_count = len(
    df[df["Event"] == "High Risk"]
)

fall_count = len(
    df[df["Event"] == "Fall Detected"]
)

running_count = len(
    df[df["Event"] == "Running"]
)

restricted_count = len(
    df[df["Event"] == "Restricted Area"]
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "🚨 High Risk",
    high_risk_count
)

col2.metric(
    "🛌 Falls",
    fall_count
)

col3.metric(
    "🏃 Running",
    running_count
)

col4.metric(
    "⛔ Restricted",
    restricted_count
)

st.divider()

# ==========================================
# ALERT TABLE
# ==========================================

st.subheader("📋 Alert History")

st.dataframe(
    df,
    use_container_width=True
)

st.divider()

# # ==========================================
# # LATEST ALERTS
# # ==========================================

# st.subheader("🕒 Latest Alerts")

# df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# latest_df = (
#     df.sort_values(
#         "Timestamp",
#         ascending=False
#     )
#     .head(10)
# )

# st.dataframe(
#     latest_df,
#     use_container_width=True
# )

# st.divider()

# ==========================================
# SNAPSHOT GALLERY
# ==========================================

st.subheader("📸 Snapshot Gallery")

snapshot_folder = "data/snapshots"

if os.path.exists(snapshot_folder):

    images = []

    for file in os.listdir(snapshot_folder):

        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            images.append(
                os.path.join(snapshot_folder, file)
            )

    images.sort(
        key=os.path.getmtime,
        reverse=True
    )

    latest_images = images[:12]

    cols = st.columns(3)

    for i, img_path in enumerate(latest_images):

        with cols[i % 3]:

            try:
                st.image(
                    Image.open(img_path),
                    caption=os.path.basename(img_path),
                    use_container_width=True
                )
            except Exception:
                pass

else:

    st.warning("No snapshots found")

st.divider()

# ==========================================
# EVENT BREAKDOWN
# ==========================================

st.subheader("📊 Event Breakdown")

event_counts = df["Event"].value_counts()

st.bar_chart(event_counts)

# ==========================================
# SURVEILLANCE ASSISTANT
# ==========================================

st.divider()
st.subheader("🔍 Surveillance Assistant")

# Show previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input(
    "Ask about surveillance incidents..."
)

if question:

    # Show user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)

    with st.spinner("Searching incidents..."):

        answer, image_paths = process_query(
            question,
            df
        )

    # Save assistant reply
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.chat_message("assistant"):
        st.text(answer)


        if len(image_paths) > 0:

            st.write("### 📸 Related Images")

            cols = st.columns(3)

            for i, path in enumerate(image_paths):

                path = path.replace("\\", "/")

                if os.path.exists(path):

                    with cols[i % 3]:

                        st.image(
                            path,
                            use_container_width=True
                        )
                


# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<hr>

<center>
AI Surveillance Dashboard • Powered by YOLOv8, TorchReID, OpenCV & Streamlit
</center>
""", unsafe_allow_html=True)