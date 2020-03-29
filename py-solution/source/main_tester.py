import pytest
import source.data_management.data_manager as data_manager

def inc(x):
    return x + 1

def test_answer():
    assert data_manager.get_training_data() is not None
