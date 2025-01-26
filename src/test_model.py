import pytest
from model import MyModel  # Import your model here

def test_model():
    model = MyModel()
    assert model.predict([1, 2, 3]) == [1]  # Example test
