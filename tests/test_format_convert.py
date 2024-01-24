import pytest
from bactrack import io

def test_hiers_to_df_basic():
    # Test basic functionality
    input_data = ...
    expected_output = ...
    assert io.hiers_to_df(input_data) == expected_output

def test_hiers_to_df_edge_case():
    # Test an edge case
    ...

def test_hiers_to_df_error_handling():
    # Test error handling
    with pytest.raises(SomeException):
        io.hiers_to_df(invalid_input)
