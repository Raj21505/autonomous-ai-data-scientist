# backend/state.py
# Store datasets, models, and their categorical encoding mappings
# Structure: { dataset_id: { "original": df, "cleaned": df, "categorical_mapping": {...}, "model_trainer": trainer_obj } }
datasets = {}
# Structure: { dataset_id: { "target": str, "trainer": ModelTrainer, "results": dict } }
trained_models = {}