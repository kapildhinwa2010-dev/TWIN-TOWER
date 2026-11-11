import numpy as np
import pytest

from twin_tower.scan import duplication_report, similarity_matrix


def _stack():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(8, 8))
    b = rng.normal(size=(8, 8))
    c = rng.normal(size=(8, 8))
    return [a, a * 2.0, b, b * 0.5, c, c.copy()]


def test_similarity_matrix():
    sim = similarity_matrix(_stack())
    assert sim.shape == (6, 6)
    assert np.allclose(np.diag(sim), 1.0)
    assert sim[0, 1] == pytest.approx(1.0)
    assert sim[2, 3] == pytest.approx(1.0)
    assert sim[4, 5] == pytest.approx(1.0)
    assert sim[0, 2] < 0.5
    assert sim == pytest.approx(sim.T)


def test_duplication_report():
    r = duplication_report(_stack(), cos_tol=0.9)
    assert r["num_layers"] == 6
    assert r["clusters"] == [[0, 1], [2, 3], [4, 5]]
    assert r["num_towers"] == 3
    assert r["duplicate_layers"] == 3
    assert r["redundancy_rate"] == pytest.approx(0.5)


def test_most_similar_pair():
    r = duplication_report(_stack())
    pair = r["most_similar_pair"]
    assert pair[0] == 0 and pair[1] == 1
    assert pair[2] == pytest.approx(1.0)


def test_all_distinct():
    rng = np.random.default_rng(3)
    stack = [rng.normal(size=(6, 6)) for _ in range(4)]
    r = duplication_report(stack, cos_tol=0.99)
    assert r["clusters"] == [[0], [1], [2], [3]]
    assert r["duplicate_layers"] == 0
    assert r["redundancy_rate"] == 0.0


def test_single_layer():
    rng = np.random.default_rng(1)
    r = duplication_report([rng.normal(size=(4, 4))])
    assert r["clusters"] == [[0]]
    assert r["most_similar_pair"] is None


def test_errors():
    with pytest.raises(ValueError):
        similarity_matrix([])
    with pytest.raises(ValueError):
        similarity_matrix([np.eye(4), np.ones((3, 3))])
    with pytest.raises(ValueError):
        similarity_matrix([np.zeros(4)])