"""Request-local wall-clock spans. Nested components are reported separately."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import time

_active=ContextVar('raccoon_timing',default=None)


def snapshot():
    current=_active.get()
    if current is None:return {}
    return {current['component']:{'elapsed_seconds':time.perf_counter()-current['started'],
        'spans_seconds':dict(current['spans']),'span_calls':dict(current['calls']),
        'clock':'perf_counter','nested_spans_are_not_additive':True}}


@contextmanager
def span(name):
    current=_active.get();began=time.perf_counter()
    try:yield
    finally:
        if current is not None:
            current['spans'][name]=current['spans'].get(name,0)+time.perf_counter()-began
            current['calls'][name]=current['calls'].get(name,0)+1


def measured(component):
    def decorate(function):
        @wraps(function)
        def call(*args,**kwargs):
            token=_active.set({'component':component,'started':time.perf_counter(),'spans':{},'calls':{}})
            try:
                result=function(*args,**kwargs)
                if isinstance(result,dict):result={**result,'timings':{**result.get('timings',{}),**snapshot()}}
                return result
            finally:_active.reset(token)
        return call
    return decorate
