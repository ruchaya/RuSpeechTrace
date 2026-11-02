from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
BASE = Path("results/raw/shap_ruasd_5seeds_100perclass")
INPUT = BASE / "top20_features_by_generator.csv"

OUT_PNG = BASE / "shap_generator_panels_top5.png"
OUT_SVG = BASE / "shap_generator_panels_top5.svg"
OUT_CSV = BASE / "shap_generator_panels_top5_used.csv"

# 8 representative generators for the paper
GENERATORS = [
    "mms_tts_rus",
    "tts_vits_ru_hf",
    "bark_small_tts",
    "coquiTTS",
    "g2pTTS",
    "tts_ru_hifigan_ruslan",
    "espeech_rl-v1",
    "voskTTS",
]

TOP_K = 5


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def pretty_generator(name: str) -> str:
    mapping = {
        "mms_tts_rus": "MMS-TTS",
        "tts_vits_ru_hf": "tts_vits_ru_hf",
        "bark_small_tts": "Bark",
        "coquiTTS": "coquiTTS",
        "g2pTTS": "g2pTTS",
        "tts_ru_hifigan_ruslan": "tts_ru_hifigan_ruslan",
        "espeech_rl-v1": "espeech_rl-v1",
        "voskTTS": "voskTTS",
    }
    return mapping.get(name, name)


def short_feature_name(x: str) -> str:
    x = str(x)
    x = x.replace("spectral_", "")
    x = x.replace("prosody_", "")
    x = x.replace("_delta2_", "_d2_")
    x = x.replace("_delta_", "_d1_")
    return x


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)

    required = {
        "generator",
        "rank",
        "feature",
        "mean_abs_shap",
        "sd_abs_shap",
        "share_percent_mean",
        "share_percent_sd",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    sub = df[df["generator"].isin(GENERATORS)].copy()
    if sub.empty:
        raise RuntimeError("No rows found for selected generators.")

    sub = sub[sub["rank"] <= TOP_K].copy()

    missing_generators = [g for g in GENERATORS if g not in set(sub["generator"])]
    if missing_generators:
        raise RuntimeError(f"Missing generators in CSV: {missing_generators}")

    # preserve generator order
    sub["generator"] = pd.Categorical(sub["generator"], categories=GENERATORS, ordered=True)
    sub = sub.sort_values(["generator", "rank"]).copy()

    # save exact subset used in the figure
    sub.to_csv(OUT_CSV, index=False)

    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    axes = axes.ravel()

    global_max = float(np.ceil((sub["share_percent_mean"] + sub["share_percent_sd"]).max() / 5.0) * 5.0)
    global_max = max(global_max, 10.0)

    for ax, gen in zip(axes, GENERATORS):
        g = sub[sub["generator"] == gen].copy()
        g = g.sort_values("share_percent_mean", ascending=True)

        y = np.arange(len(g))
        means = g["share_percent_mean"].to_numpy(dtype=float)
        errs = g["share_percent_sd"].to_numpy(dtype=float)
        labels = [short_feature_name(x) for x in g["feature"]]

        ax.barh(y, means, xerr=errs, capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlim(0, global_max)
        ax.set_title(pretty_generator(gen), fontsize=13)
        ax.set_xlabel("Share of class-conditional |SHAP| (%)", fontsize=10)
        ax.grid(axis="x", alpha=0.3)

        # numeric labels at bar ends
        for yi, m in enumerate(means):
            ax.text(m + global_max * 0.01, yi, f"{m:.1f}", va="center", fontsize=9)

    fig.suptitle(
        "RuASD: class-conditional SHAP fingerprints for selected generators\n"
        "(five seeds, 100 test recordings per generator, top-5 features per generator)",
        fontsize=16,
        y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.985])

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_SVG, bbox_inches="tight")
    plt.close(fig)

    print("FIGURE SAVED:")
    print(OUT_PNG)
    print(OUT_SVG)
    print("TABLE OF USED VALUES SAVED:")
    print(OUT_CSV)


if __name__ == "__main__":
    main()
