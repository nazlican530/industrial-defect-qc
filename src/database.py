from pymongo import MongoClient
import os
from dotenv import load_dotenv


#  MongoDB CONNECTION

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
client = MongoClient(MONGODB_URI)


# DATABASE

db = client["industrial_defect_db"]


#  COLLECTIONS

predictions_collection = db["predictions"]
users_collection = db["users"]
