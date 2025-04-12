from pymongo import MongoClient

class MongoDBService:
    def __init__(self, uri: str, db_name: str):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

    def insert(self, collection_name: str, data: dict):
        collection = self.db[collection_name]
        result = collection.insert_one(data)
        return result.inserted_id

    def find(self, collection_name: str, query: dict):  
        collection = self.db[collection_name]
        return collection.find_one(query)