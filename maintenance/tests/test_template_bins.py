"""The template's bins of BERV: each holds files of nearly the same BERV, in
any order of the files."""
import numpy as np

from lbl.recipes.lbl_template import berv_bins


def test_a_bin_holds_files_of_nearly_the_same_berv():
    berv = np.random.default_rng(1).uniform(-27.0, 27.0, 800)
    bins = berv_bins(berv, 19)
    assert sorted(set(bins)) == list(range(19))
    assert np.bincount(bins).min() >= 800 // 19
    # adjacent ranks in one bin: the bins do not overlap in BERV
    edges = [(berv[bins == i].min(), berv[bins == i].max()) for i in range(19)]
    assert all(edges[i][1] <= edges[i + 1][0] for i in range(18))


def test_the_bins_do_not_depend_on_the_order_of_the_files():
    berv = np.random.default_rng(2).uniform(-27.0, 27.0, 300)
    order = np.random.default_rng(3).permutation(300)
    assert np.array_equal(berv_bins(berv[order], 19), berv_bins(berv, 19)[order])


def test_one_file_per_bin_up_to_the_number_of_bins():
    berv = np.array([3.0, -1.0, 7.0])
    assert sorted(berv_bins(berv, 3)) == [0, 1, 2]
