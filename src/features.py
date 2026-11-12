"""The same historical cutoff and summaries used in the history notebook."""
import numpy as np
import pandas as pd

from src.config import GROUP_KEYS, HISTORY_COLUMNS, SCORE_RELEASE_LAG


def summarize_available_scores(scores, available_days, prediction_day):
    """Arrays must be ordered by availability, deadline, then assessment ID."""
    if not len(scores):
        return [0.0] + [np.nan] * (len(HISTORY_COLUMNS) - 1)
    latest_day = available_days[-1]
    latest = scores[available_days == latest_day]
    earlier = scores[available_days < latest_day]
    previous = (earlier[available_days[:len(earlier)] == available_days[len(earlier) - 1]]
                if len(earlier) else np.array([]))
    return [float(len(scores)), scores.mean(), scores.std(ddof=0), scores.min(),
            scores.max(), latest.mean(), scores[-3:].mean(),
            latest.mean() - previous.mean() if len(previous) else 0.0,
            prediction_day - latest_day]


def features_from_records(records, prediction_day, lag_days=SCORE_RELEASE_LAG):
    """Return features and a count of records too recent to use. No target is supplied."""
    ordered = sorted(records, key=lambda r: (
        max(r['deadline'], r['date_submitted']) + lag_days,
        r['deadline'], r['id_assessment']))
    eligible = [r for r in ordered
                if max(r['deadline'], r['date_submitted']) + lag_days < prediction_day]
    scores = np.asarray([r['score'] for r in eligible], dtype=float)
    available = np.asarray([max(r['deadline'], r['date_submitted']) + lag_days
                            for r in eligible], dtype=float)
    values = summarize_available_scores(scores, available, prediction_day)
    return dict(zip(HISTORY_COLUMNS, values)), len(ordered) - len(eligible)


def build_history_features(data, lag_days=SCORE_RELEASE_LAG):
    """Build features independently within student/module/presentation groups.

    Grade release is approximated as max(deadline, submission) + lag. Unknown
    deadlines must already have been excluded. Held-out students' previously
    available scores are legitimate sequential inputs, never fitting targets.
    """
    assert data['date'].notna().all() and data['date_submitted'].notna().all()
    assert data.index.is_unique
    output = np.full((len(data), len(HISTORY_COLUMNS)), np.nan)
    positions = pd.Series(np.arange(len(data)), index=data.index)
    for _, group in data.groupby(GROUP_KEYS, sort=False):
        days = group['date'].to_numpy(dtype=float)
        submitted = group['date_submitted'].to_numpy(dtype=float)
        ids = group['id_assessment'].to_numpy()
        availability = np.maximum(days, submitted) + lag_days
        order = np.lexsort((ids, days, availability))
        source_days, source_ids = days[order], ids[order]
        source_available = availability[order]
        scores = group['score'].to_numpy(dtype=float)[order]
        for index, assessment_id, cutoff in zip(group.index, ids, days - 1):
            stop = np.searchsorted(source_available, cutoff, side='left')
            assert assessment_id not in source_ids[:stop]
            assert (source_days[:stop] < cutoff).all()
            output[positions.loc[index]] = summarize_available_scores(
                scores[:stop], source_available[:stop], cutoff)
    return pd.DataFrame(output, index=data.index, columns=HISTORY_COLUMNS)
