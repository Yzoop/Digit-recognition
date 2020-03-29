import pytest
import data_manager

def test_none_get_training_data():
    """
    This test function
    tests get_training_data() function from data_manager
    """
    assert data_manager.get_training_data() != None