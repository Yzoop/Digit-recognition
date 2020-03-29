import source.data_management.data_manager as data_manager

if __name__ == "__main__":
    print("Hello, digits!")
    if data_manager.get_training_data() is not None:
        print("wow, not empty!")