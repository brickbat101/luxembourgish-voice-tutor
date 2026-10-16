"""The own FSRS-6 scheduler against values py-fsrs 6.3.2 gave without fuzz."""

import random
from datetime import datetime, timedelta, timezone

import pytest

from app import fsrs
from app.fsrs import AGAIN, EASY, GOOD, HARD, Card

T0 = datetime(2026, 10, 15, 9, 0, tzinfo=timezone.utc)


def at(text: str) -> datetime:
    return datetime.fromisoformat(text)


# (rating, time, state, step, stability, difficulty, due)
PINNED = [
    (GOOD, T0, "learning", 1, 2.3065, 2.118103970459016, T0 + timedelta(minutes=10)),
    (GOOD, at("2026-10-15T09:10:00+00:00"), "review", None, 2.3065, 2.111214235785395,
     at("2026-10-17T09:10:00+00:00")),
    (GOOD, at("2026-10-17T09:10:00+00:00"), "review", None, 10.971048263078135, 2.1043313908464483,
     at("2026-10-28T09:10:00+00:00")),
    (AGAIN, at("2026-10-28T09:10:00+00:00"), "relearning", 0, 1.53901253028147, 7.389975788014609,
     at("2026-10-28T09:20:00+00:00")),
    (GOOD, at("2026-10-28T09:20:00+00:00"), "review", None, 1.5718415897918614, 7.377814181523433,
     at("2026-10-30T09:20:00+00:00")),
    (HARD, at("2026-11-02T09:20:00+00:00"), "review", None, 5.060102568683025, 8.244499229132952,
     at("2026-11-07T09:20:00+00:00")),
    (EASY, at("2026-11-07T09:20:00+00:00"), "review", None, 15.707568115181997, 7.643121047141945,
     at("2026-11-23T09:20:00+00:00")),
    (GOOD, at("2026-11-22T09:20:00+00:00"), "review", None, 33.07711761294955, 7.630706295391642,
     at("2026-12-25T09:20:00+00:00")),
]  # fmt: skip
# The retrievability just before the review of the step with this index.
RECALL_BEFORE = {2: 0.9094932559773545, 3: 0.8998188390602377, 5: 0.8039107011609564}


def pinned_cards() -> list[Card]:
    cards, card = [], fsrs.new_card(T0)
    for rating, moment, *_ in PINNED:
        card = fsrs.review(card, rating, moment)
        cards.append(card)
    return cards


def test_the_constants_are_the_fsrs_6_defaults():
    assert len(fsrs.W) == 21
    assert fsrs.DECAY == -0.1542
    assert fsrs.FACTOR == pytest.approx(0.98034649, abs=1e-8)


@pytest.mark.parametrize("index", range(len(PINNED)))
def test_each_step_of_the_pinned_sequence_gives_the_py_fsrs_values(index):
    card = pinned_cards()[index]
    rating, moment, state, step, stability, difficulty, due = PINNED[index]

    assert (card.state, card.step) == (state, step)
    assert card.stability == pytest.approx(stability, abs=1e-6)
    assert card.difficulty == pytest.approx(difficulty, abs=1e-6)
    assert card.due == due
    assert card.last_review == moment


@pytest.mark.parametrize(("index", "recall"), sorted(RECALL_BEFORE.items()))
def test_the_retrievability_before_a_step_is_the_py_fsrs_value(index, recall):
    before = pinned_cards()[index - 1]

    assert fsrs.retrievability(before, PINNED[index][1]) == pytest.approx(recall, abs=1e-9)


def test_a_card_never_reviewed_has_no_retrievability():
    assert fsrs.retrievability(fsrs.new_card(T0), T0 + timedelta(days=3)) == 0.0


def test_the_interval_is_the_rounded_number_of_days():
    assert fsrs.interval_days(2.3065) == 2
    assert fsrs.interval_days(10.971048263078135) == 11
    assert fsrs.interval_days(8.2956) == 8
    assert fsrs.interval_days(0.001) == 1
    assert fsrs.interval_days(1e9) == 36500


def test_hard_on_a_new_card_stays_at_step_0_for_the_mean_of_the_two_steps():
    card = fsrs.review(fsrs.new_card(T0), HARD, T0)

    assert (card.state, card.step) == ("learning", 0)
    assert card.due == T0 + timedelta(minutes=5, seconds=30)


def test_easy_on_a_new_card_graduates_with_the_easy_stability():
    card = fsrs.review(fsrs.new_card(T0), EASY, T0)

    assert (card.state, card.step) == ("review", None)
    assert card.stability == pytest.approx(8.2956)
    assert card.due == T0 + timedelta(days=8)


def test_again_in_learning_goes_back_to_step_0():
    card = fsrs.review(fsrs.new_card(T0), GOOD, T0)
    card = fsrs.review(card, AGAIN, T0 + timedelta(minutes=10))

    assert (card.state, card.step) == ("learning", 0)
    assert card.due == T0 + timedelta(minutes=11)


def test_the_replay_of_the_pinned_ratings_ends_at_the_last_step():
    card = fsrs.replay([(rating, moment) for rating, moment, *_ in PINNED], T0)

    assert card == pinned_cards()[-1]
    assert card.due == at("2026-12-25T09:20:00+00:00")


def test_a_time_in_another_zone_counts_as_the_same_moment():
    paris = timezone(timedelta(hours=2))

    card = fsrs.review(fsrs.new_card(T0), GOOD, T0.astimezone(paris))

    assert card == fsrs.review(fsrs.new_card(T0), GOOD, T0)
    assert card.due.tzinfo == timezone.utc


@pytest.mark.parametrize("rating", [0, 5, -1, True, 3.0, "3", None])
def test_a_rating_outside_1_to_4_is_a_value_error(rating):
    with pytest.raises(ValueError, match="rating"):
        fsrs.review(fsrs.new_card(T0), rating, T0)


def test_a_time_without_zone_is_a_value_error():
    with pytest.raises(ValueError, match="time zone"):
        fsrs.review(fsrs.new_card(T0), GOOD, datetime(2026, 10, 15, 9, 0))
    with pytest.raises(ValueError, match="time zone"):
        fsrs.new_card(datetime(2026, 10, 15, 9, 0))


def test_a_reviewed_card_without_stability_is_a_value_error():
    card = Card("review", None, None, None, T0, T0)

    with pytest.raises(ValueError, match="stability"):
        fsrs.review(card, GOOD, T0)


def test_200_random_reviews_keep_difficulty_and_stability_in_bounds():
    rng = random.Random(137)
    card, now = fsrs.new_card(T0), T0
    for _ in range(200):
        now += timedelta(minutes=rng.choice([0, 1, 10, 60, 60 * 24, 60 * 24 * 3, 60 * 24 * 30, 60 * 24 * 400]))
        card = fsrs.review(card, rng.randint(AGAIN, EASY), now)

        assert 1.0 <= card.difficulty <= 10.0
        assert card.stability >= fsrs.STABILITY_MIN
        assert card.state in fsrs.STATES
        assert card.due > now
