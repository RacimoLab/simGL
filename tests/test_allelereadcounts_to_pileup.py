import pytest
import os
import numpy as np
import simGL


def test_file_is_created(arc, tmp_path):
    out = str(tmp_path / "out.pileup")
    simGL.allelereadcounts_to_pileup(arc, out)
    assert os.path.exists(out)

def test_line_count_equals_sites(arc, tmp_path):
    out = str(tmp_path / "out.pileup")
    simGL.allelereadcounts_to_pileup(arc, out)
    with open(out) as f:
        lines = f.readlines()
    assert len(lines) == arc.shape[0]

def test_each_line_has_correct_columns(arc, tmp_path):
    out = str(tmp_path / "out.pileup")
    simGL.allelereadcounts_to_pileup(arc, out)
    with open(out) as f:
        for line in f:
            fields = line.strip().split("\t")
            # 3 fixed fields + 3 per individual (depth, bases, quals)
            assert len(fields) == 3 + arc.shape[1] * 3

def test_total_reads_match_arc(arc, tmp_path):
    # The sum of read depths in the pileup must equal arc.sum()
    out = str(tmp_path / "out.pileup")
    simGL.allelereadcounts_to_pileup(arc, out)
    total_from_pileup = 0
    with open(out) as f:
        for line in f:
            fields = line.strip().split("\t")
            # depth fields are at positions 3, 6, 9, ... (every 3rd starting at 3)
            for k in range(arc.shape[1]):
                total_from_pileup += int(fields[3 + k * 3])
    assert total_from_pileup == int(arc.sum())


# --- Input validation ---

def test_bad_arc_raises(tmp_path):
    with pytest.raises(TypeError):
        simGL.allelereadcounts_to_pileup(np.zeros((5, 3)), str(tmp_path / "out.pileup"))

def test_bad_output_raises(arc):
    with pytest.raises(TypeError):
        simGL.allelereadcounts_to_pileup(arc, 123)
