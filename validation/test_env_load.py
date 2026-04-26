import os
from dotenv import load_dotenv

load_dotenv()

assert os.environ.get("FRED_API_KEY"), "FRED_API_KEY is missing or empty"
assert os.environ.get("POLYGON_API_KEY"), "POLYGON_API_KEY is missing or empty"
assert os.environ.get("NEWSAPI_KEY"), "NEWSAPI_KEY is missing or empty"

print("env load: PASS")
