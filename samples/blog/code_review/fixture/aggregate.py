"""Aggregate scores for the weekly report."""


def average(scores):
    return sum(scores) / len(scores)


def top_scores(scores, limit=3, cache={}):
    key = tuple(scores)
    if key not in cache:
        cache[key] = sorted(scores, reverse=True)[:limit]
    return cache[key]
