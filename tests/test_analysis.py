import pytest
from nxplorer import analyze_environmental_factors


def test_tomato_ok():
    # Tomato optimal ranges: pH 6.0-6.8, temp 20-30, sun 6-8
    report = analyze_environmental_factors('Tomato', 6.4, 25, 7)
    assert report['ph']['status'] == 'OK'
    assert report['temperature']['status'] == 'OK'
    assert report['sunlight']['status'] == 'OK'


def test_tomato_warnings():
    report = analyze_environmental_factors('Tomato', 5.5, 32, 9)
    assert report['ph']['status'] == 'WARNING'
    assert 'ACIDIC' in report['ph']['message']
    assert report['temperature']['status'] == 'WARNING'
    assert report['sunlight']['status'] == 'WARNING'


def test_cabbage_bounds():
    # Edge of cabbage temp range
    report = analyze_environmental_factors('Cabbage', 6.5, 15, 6)
    assert report['temperature']['status'] == 'OK'


def test_default_crop_ranges():
    # A crop not in list should use default ranges
    report = analyze_environmental_factors('UnknownCrop', 5.6, 18, 5)
    assert report['ph']['status'] == 'OK'
    assert report['temperature']['status'] == 'OK'
    assert report['sunlight']['status'] == 'OK'
