import pytest

def fahrenheit_to_celsius(fahrenheit):
    celsius = (fahrenheit - 32) * 5/9
    return celsius

def test_fahrenheit_to_celsius():
    assert fahrenheit_to_celsius(98.6) == pytest.approx(37.0, rel=1e-2)
    assert fahrenheit_to_celsius(32) == pytest.approx(0, rel=1e-2)
    assert fahrenheit_to_celsius(212) == pytest.approx(100, rel=1e-2)

# Test the function
fahrenheit = 98.6
celsius = fahrenheit_to_celsius(fahrenheit)
celsius

extra_info = {}
extra_info["total_pages"] = 3
extra_info["file_path"] = "filepath"

extra_info=dict(
    extra_info,
    **{
        "source": f"{1+1}",
    },
)