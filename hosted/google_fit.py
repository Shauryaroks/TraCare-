import os
import datetime
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

# Scopes required to read fitness activity
SCOPES = ['https://www.googleapis.com/auth/fitness.activity.read']

REDIRECT_URI = 'http://localhost:8501'  # Change to your deployed URL if needed

def authenticate_google_fit():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def get_auth_url():
    flow = Flow.from_client_secrets_file(
        'credentials.json',
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    auth_url, _ = flow.authorization_url(prompt='consent')
    return auth_url

def fetch_token(code):
    flow = Flow.from_client_secrets_file(
        'credentials.json',
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    flow.fetch_token(code=code)
    return flow.credentials

def fetch_step_count(creds, days=1):
    service = build('fitness', 'v1', credentials=creds)

    # Define the data source for step count
    data_source_id = "derived:com.google.step_count.delta:com.google.android.gms:estimated_steps"

    # Define start and end time in nanoseconds since epoch
    end_time = int(datetime.datetime.utcnow().timestamp() * 1e9)
    start_time = int((datetime.datetime.utcnow() - datetime.timedelta(days=days)).timestamp() * 1e9)

    # Fetch step count data from Google Fit
    dataset = f"{start_time}-{end_time}"
    dataset_response = service.users().dataSources(). \
        datasets().get(userId='me', dataSourceId=data_source_id, datasetId=dataset).execute()

    total_steps = 0
    points = dataset_response.get('point', [])
    if not points:
        return 0
    for point in points:
        for field in point.get('value', []):
            total_steps += field.get('intVal', 0)

    return total_steps

if __name__ == "__main__":
    creds = authenticate_google_fit()
    if creds:
        steps = fetch_step_count(creds)
        print(f"Steps in last 24 hours: {steps}")
    else:
        print("Run the Streamlit app for authentication.")
