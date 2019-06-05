from load_data_collision import load_data


data_train, data_val, data_test = load_data(
    'collision_dataset_classification', for_classification=True, num_files=10, num_samples=1000, to_log=True)
