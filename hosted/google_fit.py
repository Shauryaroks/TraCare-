import os
import datetime
import streamlit as st
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

# Scopes required to read fitness activity
SCOPES = ['https://www.googleapis.com/auth/fitness.activity.read']

# Get redirect URI from secrets or use default
REDIRECT_URI = st.secrets.get("GOOGLE_FIT_REDIRECT_URI", "http://localhost:8501")

def authenticate_google_fit():
    """Authenticate using stored token or return None if no valid credentials"""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                with open('token.json', 'w') as token:
                    token.write(creds.to_json())
            except Exception as e:
                st.error(f"Failed to refresh credentials: {e}")
                # Delete invalid token file
                os.remove('token.json')
                return None
    return creds

def get_auth_url():
    """Generate Google OAuth authorization URL"""
    try:
        # Create credentials from Streamlit secrets
        client_config = {
            "web": {
                "client_id": st.secrets["GOOGLE_CLIENT_ID"],
                "client_secret": st.secrets["GOOGLE_CLIENT_SECRET"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [REDIRECT_URI]
            }
        }
        
        flow = Flow.from_client_config(
            client_config,
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI
        )
        auth_url, _ = flow.authorization_url(prompt='consent')
        return auth_url
    except KeyError as e:
        st.error(f"Missing Google OAuth secret: {e}")
        st.info("Please add GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET to your Streamlit secrets.")
        return None

def fetch_token(code):
    """Exchange authorization code for access token"""
    try:
        # Create credentials from Streamlit secrets
        client_config = {
            "web": {
                "client_id": st.secrets["GOOGLE_CLIENT_ID"],
                "client_secret": st.secrets["GOOGLE_CLIENT_SECRET"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [REDIRECT_URI]
            }
        }
        
        flow = Flow.from_client_config(
            client_config,
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI
        )
        flow.fetch_token(code=code)
        return flow.credentials
    except KeyError as e:
        st.error(f"Missing Google OAuth secret: {e}")
        return None

def fetch_step_count(creds, days=1):
    """Fetch step count from Google Fit API"""
    try:
        service = build('fitness', 'v1', credentials=creds)
        
        # Define the data source for step count
        data_source_id = "derived:com.google.step_count.delta:com.google.android.gms:estimated_steps"
        
        # Define start and end time in nanoseconds since epoch
        end_time = int(datetime.datetime.utcnow().timestamp() * 1e9)
        start_time = int((datetime.datetime.utcnow() - datetime.timedelta(days=days)).timestamp() * 1e9)
        
        # Fetch step count data from Google Fit
        dataset = f"{start_time}-{end_time}"
        dataset_response = service.users().dataSources().datasets().get(
            userId='me', 
            dataSourceId=data_source_id, 
            datasetId=dataset
        ).execute()
        
        total_steps = 0
        points = dataset_response.get('point', [])
        
        if not points:
            return 0
            
        for point in points:
            for field in point.get('value', []):
                total_steps += field.get('intVal', 0)
                
        return total_steps
    except Exception as e:
        st.error(f"Error fetching step data: {e}")
        return 0

if __name__ == "__main__":
    creds = authenticate_google_fit()
    if creds:
        steps = fetch_step_count(creds)
        print(f"Steps in last 24 hours: {steps}")
    else:
        print("Run the Streamlit app for authentication.")
