from flag_gems.fused.chunk_gated_delta_rule import (
    chunk_gated_delta_rule_fwd as pkg_chunk_gated_delta_rule_fwd,
)


def test_import_chunk_gated_delta_rule_symbol():
    assert callable(pkg_chunk_gated_delta_rule_fwd)
