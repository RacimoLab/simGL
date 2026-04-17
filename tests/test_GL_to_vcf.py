import pytest
import os
import numpy as np
import simGL


@pytest.fixture
def vcf_inputs(gm, arc, ref_alt, pos, GL):
    ref, alt = ref_alt
    Ra     = simGL.ref_alt_to_index(ref, alt)
    GL_sub = simGL.normalize_GL(simGL.subset_GL(GL, Ra, ploidy=2))
    names  = [f"ind{i}" for i in range(arc.shape[1])]
    pos1   = pos + 1   # convert to 1-based
    return GL_sub, arc, ref, alt, pos1, names


# --- File structure ---

def test_file_is_created(vcf_inputs, tmp_path):
    GL_sub, arc, ref, alt, pos1, names = vcf_inputs
    out = str(tmp_path / "out.vcf")
    simGL.GL_to_vcf(GL_sub, arc, ref, alt, pos1, names, out)
    assert os.path.exists(out)

def test_header_lines_present(vcf_inputs, tmp_path):
    GL_sub, arc, ref, alt, pos1, names = vcf_inputs
    out = str(tmp_path / "out.vcf")
    simGL.GL_to_vcf(GL_sub, arc, ref, alt, pos1, names, out)
    with open(out) as f:
        content = f.read()
    assert "##fileformat=VCFv4.2" in content
    assert "##FORMAT=<ID=GT" in content
    assert "##FORMAT=<ID=GL" in content
    assert "##FORMAT=<ID=AD" in content

def test_col_header_has_sample_names(vcf_inputs, tmp_path):
    GL_sub, arc, ref, alt, pos1, names = vcf_inputs
    out = str(tmp_path / "out.vcf")
    simGL.GL_to_vcf(GL_sub, arc, ref, alt, pos1, names, out)
    with open(out) as f:
        for line in f:
            if line.startswith("#CHROM"):
                for name in names:
                    assert name in line
                break

def test_data_row_count(vcf_inputs, tmp_path):
    GL_sub, arc, ref, alt, pos1, names = vcf_inputs
    out = str(tmp_path / "out.vcf")
    simGL.GL_to_vcf(GL_sub, arc, ref, alt, pos1, names, out)
    with open(out) as f:
        data_lines = [l for l in f if not l.startswith("#")]
    assert len(data_lines) == GL_sub.shape[0]

def test_contig_header_written_when_provided(vcf_inputs, tmp_path):
    GL_sub, arc, ref, alt, pos1, names = vcf_inputs
    out = str(tmp_path / "out.vcf")
    simGL.GL_to_vcf(GL_sub, arc, ref, alt, pos1, names, out, contig_length=100_000)
    with open(out) as f:
        content = f.read()
    assert "##contig" in content


# --- GT calls ---

def test_gt_calls_valid(vcf_inputs, tmp_path):
    GL_sub, arc, ref, alt, pos1, names = vcf_inputs
    out = str(tmp_path / "out.vcf")
    simGL.GL_to_vcf(GL_sub, arc, ref, alt, pos1, names, out)
    valid_gts = {"0/0", "0/1", "1/1", "./."}
    with open(out) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            for sample_field in fields[9:]:
                gt = sample_field.split(":")[0]
                assert gt in valid_gts

def test_gt_hom_ref_when_no_alt_reads(tmp_path):
    # One site, one individual: all ref reads → should call 0/0
    arc = np.array([[[50, 0, 0, 0]]])          # 50 A-reads (ref=A)
    GL_full = simGL.allelereadcounts_to_GL(arc, e=0.01, ploidy=2, normalized=False)
    alleles = np.array([[0, 1]])               # ref=A, alt=C
    GL_sub  = simGL.normalize_GL(simGL.subset_GL(GL_full, alleles, ploidy=2))
    ref     = np.array(["A"])
    alt     = np.array(["C"])
    pos     = np.array([100])
    out     = str(tmp_path / "out.vcf")
    simGL.GL_to_vcf(GL_sub, arc, ref, alt, pos, ["ind0"], out)
    with open(out) as f:
        for line in f:
            if not line.startswith("#"):
                gt = line.strip().split("\t")[9].split(":")[0]
                assert gt == "0/0"

def test_gt_missing_when_no_reads(tmp_path):
    # One site, one individual: no reads → GL all 0 → should call ./.
    arc     = np.array([[[0, 0, 0, 0]]])
    GL_full = simGL.allelereadcounts_to_GL(arc, e=0.01, ploidy=2, normalized=False)
    alleles = np.array([[0, 1]])
    GL_sub  = simGL.normalize_GL(simGL.subset_GL(GL_full, alleles, ploidy=2))
    ref     = np.array(["A"])
    alt     = np.array(["C"])
    pos     = np.array([100])
    out     = str(tmp_path / "out.vcf")
    simGL.GL_to_vcf(GL_sub, arc, ref, alt, pos, ["ind0"], out)
    with open(out) as f:
        for line in f:
            if not line.startswith("#"):
                gt = line.strip().split("\t")[9].split(":")[0]
                assert gt == "./."


# --- AD field ---

def test_ad_field_sums_to_total_coverage(vcf_inputs, tmp_path):
    GL_sub, arc, ref, alt, pos1, names = vcf_inputs
    out = str(tmp_path / "out.vcf")
    simGL.GL_to_vcf(GL_sub, arc, ref, alt, pos1, names, out)
    with open(out) as f:
        site_idx = 0
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            for j, sample_field in enumerate(fields[9:]):
                ad = list(map(int, sample_field.split(":")[2].split(",")))
                assert sum(ad) == int(arc[site_idx, j].sum())
            site_idx += 1


# --- Input validation ---

def test_bad_GL_raises(arc, ref_alt, pos, tmp_path):
    ref, alt = ref_alt
    with pytest.raises(TypeError):
        simGL.GL_to_vcf(np.zeros((5, 3)), arc, ref, alt, pos, ["ind0"], str(tmp_path / "out.vcf"))

def test_mismatched_sites_raises(vcf_inputs, tmp_path):
    GL_sub, arc, ref, alt, pos1, names = vcf_inputs
    with pytest.raises(TypeError):
        simGL.GL_to_vcf(GL_sub[:-1], arc, ref, alt, pos1, names, str(tmp_path / "out.vcf"))

def test_wrong_sample_names_length_raises(vcf_inputs, tmp_path):
    GL_sub, arc, ref, alt, pos1, names = vcf_inputs
    with pytest.raises(TypeError):
        simGL.GL_to_vcf(GL_sub, arc, ref, alt, pos1, names[:-1], str(tmp_path / "out.vcf"))
