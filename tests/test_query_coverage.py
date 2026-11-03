"""Offline query coverage must include every depth of a mixed leaf partition."""
import itertools
import numpy as np
from reference.cells import widths_from_refinement
from reference.queries import sample_leaf_cells


def test_mixed_partition_leaf_coverage():
    refined = [{(0,0,0)}, {(0,0,0)}, set()]
    cells = sample_leaf_cells(refined,41)
    expected = {(1,c) for c in itertools.product(range(2),repeat=3) if any(c)}
    expected.update((2,c) for c in itertools.product(range(2),repeat=3))
    assert set(cells) == expected
    assert cells == sample_leaf_cells(refined,41)
    centers = np.array([(np.asarray(c)+0.5)*2.**-d for d,c in cells])
    np.testing.assert_array_equal(widths_from_refinement(refined,centers),[2.**-d for d,_ in cells])


def test_leaf_selection_is_bounded_per_depth():
    refined = [set(itertools.product(range(1<<d),repeat=3)) for d in range(4)]+[set()]
    cells = sample_leaf_cells(refined,72,per_depth=32)
    assert len(cells) == 32
    assert {d for d,_ in cells} == {4}
    assert len(set(cells)) == 32
