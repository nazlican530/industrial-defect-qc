from pymongo import MongoClient


#  MongoDB CONNECTION

client = MongoClient("mongodb://localhost:27017")


# DATABASE

db = client["industrial_defect_db"]


#  COLLECTIONS

predictions_collection = db["predictions"]
users_collection = db["users"]
