import pytest
import numpy as np
import datetime

from preprocessing import decide_whether_summer_time

def test_dates():
    assert decide_whether_summer_time(datetime.datetime(year=2022, month=10, day=29)) == True
    assert decide_whether_summer_time(datetime.datetime(year=2022, month=10, day=30)) == False
    assert decide_whether_summer_time(datetime.datetime(year=2023, month=10, day=28)) == True
    assert decide_whether_summer_time(datetime.datetime(year=2023, month=10, day=29)) == False
    assert decide_whether_summer_time(datetime.datetime(year=2022, month=6, day=29)) == True
    assert decide_whether_summer_time(datetime.datetime(year=2022, month=2, day=29)) == False
    assert decide_whether_summer_time(datetime.datetime(year=2022, month=3, day=26)) == False
    assert decide_whether_summer_time(datetime.datetime(year=2022, month=3, day=27)) == True
    assert decide_whether_summer_time(datetime.datetime(year=2023, month=3, day=25)) == False
    assert decide_whether_summer_time(datetime.datetime(year=2023, month=3, day=26)) == True


