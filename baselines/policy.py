from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InputPolicy:
    baseline_key: str
    allowed_input_sets: tuple[tuple[str, ...], ...]
    target: str
    compare_waveform_metrics: bool
    compare_scalar_metrics: bool
    keep_upstream_code_untouched: bool = True


INPUT_POLICIES = {
    "ptt_based": InputPolicy(
        baseline_key="ptt_based",
        allowed_input_sets=(("II", "PLETH"),),
        target="ABP",
        compare_waveform_metrics=False,
        compare_scalar_metrics=True,
    ),
    "ppg2abp": InputPolicy(
        baseline_key="ppg2abp",
        allowed_input_sets=(("PLETH",),),
        target="ABP",
        compare_waveform_metrics=True,
        compare_scalar_metrics=True,
    ),
    "inn_par": InputPolicy(
        baseline_key="inn_par",
        allowed_input_sets=(("PLETH",),),
        target="ABP",
        compare_waveform_metrics=True,
        compare_scalar_metrics=True,
    ),
    "papagei": InputPolicy(
        baseline_key="papagei",
        allowed_input_sets=(("PLETH",),),
        target="ABP",
        compare_waveform_metrics=False,
        compare_scalar_metrics=True,
    ),
    "unicardio": InputPolicy(
        baseline_key="unicardio",
        allowed_input_sets=(("PLETH", "II"),),
        target="ABP",
        compare_waveform_metrics=True,
        compare_scalar_metrics=True,
    ),
}
