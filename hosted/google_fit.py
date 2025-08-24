import datetime
import tomli
import tomli_w
import os
import json
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

# --------------------------
# Load credentials from TOML
# --------------------------


SCOPES = ['https://www.googleapis.com/auth/fitness.activity.read']

GOOGLE_WEB = 
REDIRECT_URI = GOOGLE_WEB["redirect_uris"][0]  # use first redirect URI

# --------------------------
# Authentication Functions
# --------------------------
def authenticate_google_fit():
    """
    Load existing credentials from TOML token section if available,
    refresh if expired. Returns Credentials object or None.
    """
    token_data = config.get("TOKEN", {})
    creds = None
    if token_data:
        creds = Credentials(
            token=token_data.get("token"),
            refresh_token=token_data.get("refresh_token"),
            token_uri=token_data.get("token_uri"),
            client_id=token_data.get("client_id"),
            client_secret=token_data.get("client_secret"),
            scopes=token_data.get("scopes"),
        )

    # Refresh if expired
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        # Save refreshed token back to TOML
        config["TOKEN"] = {
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": creds.scopes,
            "expiry": str(creds.expiry),
        }
        save_config(config)
    return creds

def get_auth_url():
    """
    Returns the Google Fit OAuth authorization URL.
    """
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": GOOGLE_WEB["client_id"],
                "project_id": GOOGLE_WEB.get("project_id", ""),
                "auth_uri": GOOGLE_WEB["auth_uri"],
                "token_uri": GOOGLE_WEB["token_uri"],
                "auth_provider_x509_cert_url": GOOGLE_WEB["auth_provider_x509_cert_url"],
                "client_secret": GOOGLE_WEB["client_secret"],
                "redirect_uris": GOOGLE_WEB["redirect_uris"],
                "javascript_origins": GOOGLE_WEB["javascript_origins"]
            }
        },
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    auth_url, _ = flow.authorization_url(prompt='consent')
    return auth_url, flow

def fetch_token(flow, code):
    """
    Exchange authorization code for credentials and save to TOML.
    """
    flow.fetch_token(code=code)
    creds = flow.credentials
    # Save token to TOML
    config["TOKEN"] = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
        "expiry": str(creds.expiry),
    }
    save_config(config)
    return creds

# --------------------------
# Fetch Step Count
# --------------------------
def fetch_step_count(creds, days=1):
    """
    Fetch step count from Google Fit for the last 'days' days.
    Returns integer total steps.
    """
    service = build('fitness', 'v1', credentials=creds)

    data_source_id = "derived:com.google.step_count.delta:com.google.android.gms:estimated_steps"

    end_time = int(datetime.datetime.utcnow().timestamp() * 1e9)
    start_time = int((datetime.datetime.utcnow() - datetime.timedelta(days=days)).timestamp() * 1e9)

    dataset_id = f"{start_time}-{end_time}"
    response = service.users().dataSources().datasets().get(
        userId='me',
        dataSourceId=data_source_id,
        datasetId=dataset_id
    ).execute()

    total_steps = 0
    points = response.get('point', [])
    for point in points:
        for field in point.get('value', []):
            total_steps += field.get('intVal', 0)
    return total_steps

# --------------------------
# Example Usage
# --------------------------
if __name__ == "__main__":
    creds = authenticate_google_fit()
    if creds:
        steps = fetch_step_count(creds)
        print(f"Steps in last 24 hours: {steps}")
    else:
        print("No credentials found. Run the Streamlit app and authorize Google Fit first.")
