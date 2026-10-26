import pandas as pd
import re
from datetime import datetime


def extract_snapshot_time(path):

    if pd.isna(path):
        return pd.NaT

    match = re.search(
        r'(\d{8}_\d{6})',
        str(path)
    )

    if match:

        return datetime.strptime(
            match.group(1),
            "%Y%m%d_%H%M%S"
        )

    return pd.NaT


# ==========================================
# EVENT MAP
# ==========================================

EVENTS = {
    "high risk": "High Risk",
    "fall": "Fall Detected",
    "falls": "Fall Detected",
    "running": "Running",
    "restricted": "Restricted Area",
    "restricted area": "Restricted Area",
    "loitering": "Global Loitering",
    "entrance": "Entrance"
}


# ==========================================
# DETECT EVENT
# ==========================================

def detect_event(question):

    question = question.lower()

    for key, value in EVENTS.items():

        if key in question:
            return value

    return None


# ==========================================
# IMAGE REQUEST
# ==========================================

def wants_images(question):

    keywords = [
        "image",
        "images",
        "photo",
        "photos",
        "picture",
        "pictures",
        "snapshot",
        "snapshots"
    ]

    question = question.lower()

    return any(
        word in question
        for word in keywords
    )


# ==========================================
# LATEST REQUEST
# ==========================================

def wants_latest(question):

    question = question.lower()

    keywords = [
        "latest",
        "recent",
        "last alert",
        "last incident"
    ]

    return any(
        word in question
        for word in keywords
    )


# ==========================================
# SUMMARY REQUEST
# ==========================================

def wants_summary(question):

    question = question.lower()

    keywords = [
        "summary",
        "report",
        "overview",
        "today"
    ]

    return any(
        word in question
        for word in keywords
    )


# ==========================================
# DATE REQUEST
# ==========================================

def get_requested_date(question):

    question = question.lower().strip()

    # today
    if "today" in question:
        return pd.Timestamp.today().date()

    # any date format yyyy-mm-dd
    match = re.search(
        r"(\d{4}-\d{2}-\d{2})",
        question
    )

    if match:

        try:

            return pd.to_datetime(
                match.group(1)
            ).date()

        except:

            return None

    return None


# ==========================================
# TIME WINDOW
# ==========================================

def get_time_window(question):

    question = question.lower()

    minute_match = re.search(
        r"last\s+(\d+)\s*(minute|minutes|min)",
        question
    )

    if minute_match:

        return int(
            minute_match.group(1)
        )

    hour_match = re.search(
        r"last\s+(\d+)\s*(hour|hours|hr)",
        question
    )

    if hour_match:

        return (
            int(hour_match.group(1))
            * 60
        )

    return None


# ==========================================
# BUILD SUMMARY
# ==========================================

def build_summary(df):

    event_counts = (
        df["Event"]
        .value_counts()
        .to_dict()
    )

    lines = []

    for event, count in event_counts.items():

        lines.append(
            f"• {event}: {count}"
        )

    return "\n".join(lines)


# ==========================================
# GET UNIQUE IMAGES
# ==========================================

def get_images(df):

    if "Snapshot" not in df.columns:

        return []

    snapshots = df[
        df["Snapshot"].notna()
    ]

    return (
        snapshots["Snapshot"]
        .drop_duplicates()
        .tolist()
    )


# ==========================================
# MAIN ENGINE
# ==========================================

def process_query(question, df):

    working_df = df.copy()

    image_paths = []

    # ======================================
    # TIMESTAMP FIX
    # ======================================

    working_df["Timestamp"] = pd.to_datetime(
        working_df["Timestamp"],
        errors="coerce"
    )

    working_df = working_df.dropna(
        subset=["Timestamp"]
    )

    working_df = working_df.sort_values(
        "Timestamp",
        ascending=False
    )


    # ======================================
    # EVENT FILTER
    # ======================================

    event_name = detect_event(question)

    requested_date = get_requested_date(
        question
    )

    if event_name:

        working_df = working_df[
            working_df["Event"] == event_name
        ]

    # ======================================
    # DATE FILTER
    # ======================================

    requested_date = get_requested_date(
        question
    )


    if requested_date:

        working_df = working_df[
            working_df["Timestamp"].dt.date
            == requested_date
        ]


    # ======================================
    # TIME FILTER
    # ======================================

    minutes = get_time_window(question)

    if minutes:

        latest_time = (
            working_df["Timestamp"].max()
        )

        cutoff_time = (
            latest_time -
            pd.Timedelta(minutes=minutes)
        )

        working_df = working_df[
            working_df["Timestamp"] >= cutoff_time
        ]

    # ======================================
    # NO DATA
    # ======================================

    if len(working_df) == 0:

        if requested_date:

            return (
                f"No activity found on {requested_date}.",
                []
            )

        return (
            "No incidents found.",
            []
        )
        
    # ======================================
    # LATEST INCIDENT IMAGE
    # ======================================

    if (
        wants_latest(question)
        and
        wants_images(question)
    ):

        if "Snapshot" not in working_df.columns:

            return (
                "No image data available.",
                []
            )

        snapshots = working_df[
            working_df["Snapshot"].notna()
        ].copy()

        if len(snapshots) == 0:

            return (
                "No images found.",
                []
            )

        # Extract timestamp from filename
        snapshots["Snapshot_Time"] = (
            snapshots["Snapshot"]
            .apply(extract_snapshot_time)
        )

        # Sort by image timestamp
        snapshots = snapshots.sort_values(
            "Snapshot_Time",
            ascending=False
        )

        latest_row = snapshots.iloc[0]

        image_path = latest_row["Snapshot"]

        answer = f"""
    📸 Latest Incident Image

    Event: {latest_row['Event']}
    Master ID: {latest_row['Master_ID']}
    Time: {latest_row['Timestamp']}
    Image Time: {latest_row['Snapshot_Time']}
    """

        return answer, [image_path]
    
    # ======================================
    # LATEST INCIDENT
    # ======================================

    if wants_latest(question):

        # working_df is already filtered by event/date/time
        latest_row = working_df.iloc[0]

        title = "Latest Incident"

        if event_name:
            title = f"Latest {event_name}"

        answer = f"""
    🚨 {title}

    Event: {latest_row['Event']}
    Master ID: {latest_row['Master_ID']}
    Time: {latest_row['Timestamp']}
    """

        image_paths = []

        if (
            "Snapshot" in latest_row.index
            and
            pd.notna(latest_row["Snapshot"])
        ):
            image_paths.append(
                latest_row["Snapshot"]
            )

        return answer, image_paths

    # ======================================
    # SUMMARY + IMAGES
    # ======================================

    if (
        wants_summary(question)
        and
        wants_images(question)
    ):

        summary = build_summary(
            working_df
        )

        image_paths = get_images(
            working_df
        )

        answer = f"""
📊 Security Summary

{summary}

Total Incidents: {len(working_df)}

Images Found: {len(image_paths)}
"""

        return answer, image_paths

    # ======================================
    # SUMMARY
    # ======================================

    if wants_summary(question):

        summary = build_summary(
            working_df
        )

        image_paths = get_images(
            working_df
        )

        answer = f"""
📊 Security Summary

{summary}

Total Incidents: {len(working_df)}
"""

        return answer, image_paths

    # ======================================
    # LAST X MINUTES / HOURS
    # ======================================

    if minutes:

        summary = build_summary(
            working_df
        )

        image_paths = get_images(
            working_df
        )

        answer = f"""
⏱ Incidents In Last {minutes} Minutes

{summary}

Total Incidents: {len(working_df)}

Images Found: {len(image_paths)}
"""

        return answer, image_paths

    # ======================================
    # EVENT REPORT WITH IMAGES
    # ======================================

    if event_name:

        image_paths = get_images(
            working_df
        )

        summary = build_summary(
            working_df
        )

        answer = f"""
🚨 {event_name} Report

{summary}

Total Incidents: {len(working_df)}

Images Found: {len(image_paths)}
"""

        return answer, image_paths

    # ======================================
    # DEFAULT REPORT
    # ======================================

    image_paths = get_images(
        working_df
    )

    summary = build_summary(
        working_df
    )

    answer = f"""
📊 Incident Report

{summary}

Total Incidents: {len(working_df)}

Images Found: {len(image_paths)}
"""

    return answer, image_paths