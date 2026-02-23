import pandas as pd

BASELINE_PATH = "eval/baseline_metrics.csv"
LORA_PATH = "eval/lora_metrics.csv"


def summarize(df, label):
    cols = ["output_token_length", "readability_grade", "latency_s"]
    stats = df[cols].agg(["mean", "median"]).T
    stats.columns = [f"{label}_mean", f"{label}_median"]
    return stats


def main():
    baseline = pd.read_csv(BASELINE_PATH)
    lora = pd.read_csv(LORA_PATH)

    base_stats = summarize(baseline, "baseline")
    lora_stats = summarize(lora, "lora")
    combined = base_stats.join(lora_stats)

    combined["delta_mean"] = combined["lora_mean"] - combined["baseline_mean"]
    combined["delta_median"] = combined["lora_median"] - combined["baseline_median"]

    print("\nMetric comparison (LoRA - Baseline):")
    print(combined.round(4))


if __name__ == "__main__":
    main()
