# backend/utils/file_loader.py

import pandas as pd
from fastapi import UploadFile
from io import BytesIO


def load_csv(file: UploadFile):
    """Load uploaded file as a pandas DataFrame.

    Accepts CSV and Excel (.xls/.xlsx). Tries to read intelligently and
    falls back between formats if detection fails.
    """
    filename = getattr(file, "filename", "") or ""
    lower = filename.lower()

    # Read raw bytes (UploadFile.file is a SpooledTemporaryFile)
    data = file.file.read()
    # reset pointer for potential future reads
    try:
        file.file.seek(0)
    except Exception:
        pass

    # Try Excel first when extension suggests it
    if lower.endswith(('.xls', '.xlsx')):
        try:
            return pd.read_excel(BytesIO(data), engine="openpyxl")
        except Exception:
            # fallback to csv parsing
            try:
                return pd.read_csv(BytesIO(data))
            except Exception as e:
                raise

    # If extension unknown, try csv then excel
    try:
        return pd.read_csv(BytesIO(data))
    except Exception:
        try:
            return pd.read_excel(BytesIO(data), engine="openpyxl")
        except Exception:
            raise
