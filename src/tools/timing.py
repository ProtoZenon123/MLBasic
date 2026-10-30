"""Measure the duration of any function call."""

from time import perf_counter


def measure_time(function, *args, **kwargs):
    start = perf_counter()
    result = function(*args, **kwargs)
    return result, perf_counter() - start
