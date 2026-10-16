"""FSRS-6 scheduling with the 21 default weights, as py-fsrs 6.3.2 does it without fuzz.

A card is learning, review or relearning; a card that was never reviewed is learning at
step 0 without stability or difficulty. Ratings are 1 Again, 2 Hard, 3 Good, 4 Easy. Every
time is a timezone-aware datetime; elapsed time counts in whole days, as py-fsrs counts it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

W = (
    0.212, 1.2931, 2.3065, 8.2956, 6.4133, 0.8334, 3.0194, 0.001, 1.8722, 0.1666, 0.796,
    1.4835, 0.0614, 0.2629, 1.6483, 0.6014, 1.8729, 0.5425, 0.0912, 0.0658, 0.1542,
)  # fmt: skip
DECAY = -W[20]
FACTOR = 0.9 ** (1 / DECAY) - 1
STABILITY_MIN = 0.001
MIN_DIFFICULTY = 1.0
MAX_DIFFICULTY = 10.0
RETENTION = 0.9
LEARNING_STEPS = (timedelta(minutes=1), timedelta(minutes=10))
RELEARNING_STEPS = (timedelta(minutes=10),)
MAX_INTERVAL_DAYS = 36500
AGAIN, HARD, GOOD, EASY = 1, 2, 3, 4
STATES = ("learning", "review", "relearning")


@dataclass(frozen=True)
class Card:
    state: str  # one of STATES
    step: int | None  # the learning or relearning step; None in review
    stability: float | None
    difficulty: float | None
    due: datetime
    last_review: datetime | None


def new_card(created: datetime) -> Card:
    """A card never reviewed, due when it was made."""
    return Card("learning", 0, None, None, _utc(created), None)


def checked_rating(rating: object) -> int:
    """`rating` when it is a whole number 1..4; a ValueError otherwise."""
    if isinstance(rating, bool) or not isinstance(rating, int) or not AGAIN <= rating <= EASY:
        raise ValueError(f"a rating is 1, 2, 3 or 4, not {rating!r}")
    return rating


def retrievability(card: Card, now: datetime) -> float:
    """The chance of recall at `now`; 0 for a card never reviewed."""
    if card.last_review is None or card.stability is None:
        return 0.0
    elapsed = max(0, (_utc(now) - _utc(card.last_review)).days)
    return (1 + FACTOR * elapsed / card.stability) ** DECAY


def interval_days(stability: float, retention: float = RETENTION, max_interval_days: int = MAX_INTERVAL_DAYS) -> int:
    """Whole days until recall falls to `retention`, at least 1 and at most `max_interval_days`."""
    days = round(stability / FACTOR * (retention ** (1 / DECAY) - 1))
    return min(max(days, 1), max_interval_days)


def review(
    card: Card,
    rating: int,
    now: datetime,
    retention: float = RETENTION,
    learning_steps: tuple[timedelta, ...] = LEARNING_STEPS,
    relearning_steps: tuple[timedelta, ...] = RELEARNING_STEPS,
    max_interval_days: int = MAX_INTERVAL_DAYS,
) -> Card:
    """The card after `rating` at `now`. A ValueError for a rating outside 1..4 or a naive time."""
    rating = checked_rating(rating)
    if card.state not in STATES:
        raise ValueError(f"unknown card state {card.state!r}; known: {', '.join(STATES)}")
    now = _utc(now)
    last = _utc(card.last_review) if card.last_review is not None else None
    days = (now - last).days if last is not None else None
    same_day = days is not None and days < 1
    stability, difficulty = card.stability, card.difficulty

    if card.state == "learning" and (stability is None or difficulty is None):
        stability = _clamp_stability(W[rating - 1])
        difficulty = _clamp_difficulty(_initial_difficulty(rating))
    else:
        if stability is None or difficulty is None:
            raise ValueError(f"a card in {card.state} needs its stability and difficulty")
        if same_day:
            stability = _short_term_stability(stability, rating)
        else:
            recall = retrievability(card, now)
            stability = _next_stability(difficulty, stability, recall, rating)
        difficulty = _next_difficulty(difficulty, rating)

    def graduated() -> tuple[str, int | None, timedelta]:
        return "review", None, timedelta(days=interval_days(stability, retention, max_interval_days))

    if card.state == "review":
        if rating == AGAIN and relearning_steps:
            state, step, interval = "relearning", 0, relearning_steps[0]
        else:
            state, step, interval = graduated()
    else:
        steps = learning_steps if card.state == "learning" else relearning_steps
        step = card.step if card.step is not None else 0
        if not steps or (step >= len(steps) and rating != AGAIN):
            state, step, interval = graduated()
        elif rating == AGAIN:
            state, step, interval = card.state, 0, steps[0]
        elif rating == HARD:
            if step == 0:
                interval = steps[0] * 1.5 if len(steps) == 1 else (steps[0] + steps[1]) / 2.0
            else:
                interval = steps[step]
            state = card.state
        elif rating == GOOD and step + 1 < len(steps):
            state, step, interval = card.state, step + 1, steps[step + 1]
        else:
            state, step, interval = graduated()
    return Card(state, step, stability, difficulty, now + interval, now)


def replay(logs: Iterable[tuple[int, datetime]], created: datetime) -> Card:
    """The card after the ratings (rating, time) in order, from a new card made at `created`."""
    card = new_card(created)
    for rating, at in logs:
        card = review(card, rating, at)
    return card


def _utc(moment: datetime) -> datetime:
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError(f"the time {moment.isoformat()} has no time zone")
    return moment.astimezone(timezone.utc)


def _clamp_stability(stability: float) -> float:
    return max(stability, STABILITY_MIN)


def _clamp_difficulty(difficulty: float) -> float:
    return min(max(difficulty, MIN_DIFFICULTY), MAX_DIFFICULTY)


def _initial_difficulty(rating: int) -> float:
    return W[4] - math.e ** (W[5] * (rating - 1)) + 1


def _short_term_stability(stability: float, rating: int) -> float:
    increase = math.e ** (W[17] * (rating - 3 + W[18])) * stability ** -W[19]
    if rating >= HARD:
        increase = max(increase, 1.0)
    return _clamp_stability(stability * increase)


def _next_difficulty(difficulty: float, rating: int) -> float:
    delta = -(W[6] * (rating - 3))
    damped = difficulty + (10.0 - difficulty) * delta / 9.0
    return _clamp_difficulty(W[7] * _initial_difficulty(EASY) + (1 - W[7]) * damped)


def _next_stability(difficulty: float, stability: float, recall: float, rating: int) -> float:
    if rating == AGAIN:
        long_term = W[11] * difficulty ** -W[12] * ((stability + 1) ** W[13] - 1) * math.e ** ((1 - recall) * W[14])
        short_term = stability / math.e ** (W[17] * W[18])
        return _clamp_stability(min(long_term, short_term))
    hard_penalty = W[15] if rating == HARD else 1
    easy_bonus = W[16] if rating == EASY else 1
    grown = stability * (
        1
        + math.e ** W[8]
        * (11 - difficulty)
        * stability ** -W[9]
        * (math.e ** ((1 - recall) * W[10]) - 1)
        * hard_penalty
        * easy_bonus
    )
    return _clamp_stability(grown)
