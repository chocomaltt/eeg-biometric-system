import os.path
import dotenv

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = [dotenv.get_key(".env", "CLOUD_SCOPES")]

# The ID and range of a sample spreadsheet.
SPREADSHEET_ID = dotenv.get_key(".env", "SPREADSHEET_ID")

# Row mapping: (data_type, window_size, stride) -> row number in sheet
ROW_MAPPING = {
    ('eo', 1, 0.5): 2,
    ('eo', 1, 1): 3,
    ('eo', 1.5, 0.5): 4,
    ('eo', 1.5, 1): 5,
    ('eo', 1.5, 1.5): 6,
    ('eo', 2, 0.5): 7,
    ('eo', 2, 1): 8,
    ('eo', 2, 1.5): 9,
    ('eo', 2, 2): 10,
    ('ec', 1, 0.5): 11,
    ('ec', 1, 1): 12,
    ('ec', 1.5, 0.5): 13,
    ('ec', 1.5, 1): 14,
    ('ec', 1.5, 1.5): 15,
    ('ec', 2, 0.5): 16,
    ('ec', 2, 1): 17,
    ('ec', 2, 1.5): 18,
    ('ec', 2, 2): 19,
}


def get_credentials():
    """Get or refresh Google API credentials."""
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return creds


def log_to_sheets(data_type, window_size, stride, top1_accuracy, eer_threshold, eer, seed):
    """
    Log training results to Google Sheets.

    Args:
        data_type: 'eo' or 'ec'
        window_size: float (1, 1.5, or 2)
        stride: float (0.5, 1, 1.5, or 2)
        top1_accuracy: float (0-1)
        eer_threshold: float
        eer: float (0-1)
        seed: int, the random seed used (used as sheet/tab name)
    """
    creds = get_credentials()

    try:
        service = build("sheets", "v4", credentials=creds)
        sheet = service.spreadsheets()

        # Get the row for this configuration
        key = (data_type.lower(), float(window_size), float(stride))
        row = ROW_MAPPING.get(key)

        if row is None:
            print(f"No row mapping for config: {key}")
            return False

        # Columns: Model | Window Size | Stride | Top1 Accuracy | EER Threshold | EER
        # We update columns D, E, F (Top1 Accuracy, EER Threshold, EER)
        range_name = f"{seed}!D{row}:F{row}"

        values = [[
            f"{top1_accuracy * 100:.2f}%",
            f"{eer_threshold:.4f}",
            f"{eer * 100:.2f}%"
        ]]

        body = {"values": values}

        result = sheet.values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=range_name,
            valueInputOption="RAW",
            body=body
        ).execute()

        print(f"Updated {result.get('updatedCells')} cells in sheet '{seed}' row {row}")
        return True

    except HttpError as err:
        print(f"Google Sheets API error: {err}")
        return False


if __name__ == "__main__":
    # Test with sample data
    log_to_sheets(
        data_type="ec",
        window_size=1,
        stride=0.5,
        top1_accuracy=0.9929,
        eer_threshold=0.8404,
        eer=0.0410,
        seed=42
    )
