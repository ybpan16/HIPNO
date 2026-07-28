from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    family: str
    inputs: str
    outputs: str
    framework: str
    source_url: str
    upstream_code_policy: str
    status: str
    note: str


BASELINE_SPECS = {
    "ptt_based": BaselineSpec(
        name="PTT-based",
        family="classical",
        inputs="ECG + PPG",
        outputs="SBP, DBP, MAP",
        framework="classical regression",
        source_url="",
        upstream_code_policy="keep_original_code_untouched",
        status="launcher_implemented",
        note=(
            "Classical Moens-Korteweg-style scalar baseline. The benchmark launcher consumes ECG lead II and PPG, "
            "detects ECG R peaks and subsequent PPG foot points, fits BP = beta0 + beta1 / PTT on the calibration "
            "split, and reports scalar SBP/DBP/MAP metrics only."
        ),
    ),
    "ppg2abp": BaselineSpec(
        name="PPG2ABP",
        family="waveform",
        inputs="PPG",
        outputs="ABP waveform",
        framework="Keras / TensorFlow",
        source_url="https://github.com/nibtehaz/PPG2ABP",
        upstream_code_policy="keep_original_code_untouched",
        status="source_locked_repo_review_pending",
        note=(
            "Authoritative repo locked. The README describes the original Keras implementation, "
            "a two-stage cascaded method for PPG-to-ABP waveform translation, with a pipeline built "
            "around data_processing.py, data_handling.py, train_models.py, predict_test.py, and evaluate.py."
        ),
    ),
    "inn_par": BaselineSpec(
        name="INN-PAR",
        family="waveform",
        inputs="PPG",
        outputs="ABP waveform",
        framework="PyTorch",
        source_url="https://github.com/soumitra1992/INNPAR-PPG2ABP",
        upstream_code_policy="keep_original_code_untouched",
        status="launcher_implemented",
        note=(
            "Authoritative repo locked. The README states that INN-PAR uses invertible blocks, jointly learns "
            "PPG/PPG-gradient to ABP/ABP-gradient mappings, and adds a multi-scale convolution module inside "
            "the invertible block. The benchmark launcher calls the official PyTorch network/data/loss modules "
            "from an untouched upstream clone and prepares the original datasets/{split}/{ppg,abp} .npy layout."
        ),
    ),
    "papagei": BaselineSpec(
        name="PaPaGei",
        family="foundation_model",
        inputs="PPG",
        outputs="SBP, DBP, MAP",
        framework="PyTorch",
        source_url="https://github.com/nokia-bell-labs/papagei-foundation-model",
        upstream_code_policy="keep_original_code_untouched",
        status="launcher_implemented",
        note=(
            "Authoritative repo locked. The README provides a feature-extraction workflow using PaPaGei-S, "
            "pretrained weights from Zenodo, 10 s segmented PPG resampled to 125 Hz, and 512-dimensional embeddings "
            "from ResNet1DMoE. The benchmark launcher uses the official preprocessing/model utilities from an "
            "untouched upstream clone, freezes PaPaGei-S, and fits Ridge scalar heads for SBP, DBP, and MAP."
        ),
    ),
    "unicardio": BaselineSpec(
        name="UniCardio",
        family="waveform",
        inputs="PPG with generated ECG intermediate",
        outputs="ABP waveform",
        framework="PyTorch",
        source_url="https://github.com/thu-ml/UniCardio",
        upstream_code_policy="keep_original_code_untouched",
        status="launcher_implemented",
        note=(
            "Authoritative repo locked. The README describes UniCardio as a multi-modal diffusion Transformer "
            "for denoising, imputation, and translation across cardiovascular signals such as PPG, ECG, and BP, "
            "with official training and evaluation entrypoints in base_model/train_original.py and test_final.py. "
            "The benchmark launcher follows the official chain: PPG->ECG with model_flag='02', then ECG->BP "
            "with the MIMIC fine-tune branch and model_flag='21'."
        ),
    ),
}
