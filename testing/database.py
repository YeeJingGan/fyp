from pymongo import MongoClient
import gridfs
from bson import ObjectId

class MongoDBService:
    def __init__(self, uri: str, db_name: str):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.fs = gridfs.GridFS(self.db)

    def insert_patient(self, collection_name: str, data: dict):
        collection = self.db[collection_name]
        result = collection.insert_one(data)
        return result.inserted_id
    
    def save_file(self, file_name: str, file_data: bytes):
        file_id = self.fs.put(file_data, filename=file_name)
        return file_id
    
    def get_file(self, file_id):
        return self.fs.get(file_id).read()
    
    def find(self, collection_name: str, patient_reference: str):  
        collection = self.db[collection_name]
        return collection.find_one({"patient_reference": patient_reference})
    
    def find_patient(self, collection_name: str, patient_reference: str):
        record = self.find(collection_name, patient_reference)
            
        if record:
            modality = record.get("modality", "Unknown")   
            if "ori_mri_id" in record and "anomaly_mask_id" in record:
                ori_file_id = record["ori_mri_id"]
                ori_file_content = self.get_file(ori_file_id)

                anomaly_file_id = record["anomaly_mask_id"]
                anomaly_file_content = self.get_file(anomaly_file_id)

                return ori_file_content, anomaly_file_content, modality   
            else:
                return None, None, None
        else:
            return None, None, None

    def delete_patient(self, collection_name: str, patient_reference: str):
        collection = self.db[collection_name]
        record = self.find(collection_name, patient_reference)
        
        if record and "ori_mri_id" in record and "anomaly_mask_id" in record:
            ori_file_id = record["ori_mri_id"]
            self.fs.delete(ori_file_id)

            anomaly_file_id = record["anomaly_mask_id"]
            self.fs.delete(anomaly_file_id)

        result = collection.delete_one({"patient_reference": patient_reference})

        return result.deleted_count

