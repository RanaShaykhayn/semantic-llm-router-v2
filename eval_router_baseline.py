import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

import src.router_architecture as ra
from src.router_architecture import SemanticLLMRouter


ra.get_estimated_latency = lambda route: (None, 0)

TEST_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test.csv")
OUT_DIR = Path("reports")
OUT_DIR.mkdir(exist_ok=True)

GOLD_MAP = {
    "greeting": "local",
    "simple": "flash",
    "ambiguous": "flash",
    "complex": "pro",
    "injection": "blocked",
}
LABELS = ["local", "flash", "pro", "blocked"]


def route_to_label(router: SemanticLLMRouter, route: str) -> str:
    if route == "Local Simulator":
        return "local"
    if route == router.gemini_pro:
        return "pro"
    if route == router.gemini_flash:
        return "flash"
    return "other"  


def predict(router: SemanticLLMRouter, text: str) -> str:
    
    cleaned = router.preprocess_text(text)
    if router._is_prompt_injection(cleaned):
        return "blocked"
    return route_to_label(router, router.predict_route(text))


def main() -> None:
    df = pd.read_csv(TEST_PATH)
    df["gold"] = df["label"].map(GOLD_MAP)
    assert df["gold"].notna().all(), "unknown label in test.csv"

    router = SemanticLLMRouter()
    preds, times = [], []
    for i, text in enumerate(df["text"], start=1):
        t0 = time.perf_counter()
        try:
            preds.append(predict(router, text))
        except Exception as exc:  
            print(f"[{i}] ERROR on {text!r}: {exc}")
            preds.append("error")
        times.append((time.perf_counter() - t0) * 1000)
    df["pred"] = preds
    df["predict_ms"] = times
    df["correct"] = df["pred"] == df["gold"]

    labels = LABELS + [x for x in ("other", "error") if x in set(preds)]
    lines = []
    lines.append(f"# Baseline: current router on {TEST_PATH.name} ({len(df)} queries)\n")
    lines.append(f"Accuracy: {df['correct'].mean():.3f}   (errors: {preds.count('error')})\n")
    lines.append("## Per-route precision / recall / F1\n```")
    lines.append(classification_report(df["gold"], df["pred"], labels=LABELS,
                                       zero_division=0, digits=3))
    lines.append("```\n## Confusion matrix (rows = gold, cols = predicted)\n```")
    cm = pd.DataFrame(confusion_matrix(df["gold"], df["pred"], labels=labels),
                      index=labels, columns=labels)
    lines.append(cm.to_string())
    lines.append("```\n## Accuracy by original label")
    lines.append(df.groupby("label")["correct"].mean().round(3).to_string())
    lines.append("\n## Accuracy by language")
    lines.append(df.groupby("lang")["correct"].mean().round(3).to_string())
    lines.append("\n## Accuracy on hard cases (hard=1) vs normal (hard=0)")
    lines.append(df.groupby("hard")["correct"].mean().round(3).to_string())
    lines.append(f"\nMedian predict time: {df['predict_ms'].median():.0f} ms "
                 f"(includes the embedding API call)")

    report = "\n".join(lines)
    print(report)
    (OUT_DIR / "baseline_router.md").write_text(report, encoding="utf-8")
    df.to_csv(OUT_DIR / "baseline_router_predictions.csv", index=False)

    wrong = df[~df["correct"]][["text", "label", "pred"]]
    print(f"\n{len(wrong)} wrong predictions saved in reports/baseline_router_predictions.csv")


if __name__ == "__main__":
    main()