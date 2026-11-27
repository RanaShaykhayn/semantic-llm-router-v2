import re
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "reports"
OUT_DIR.mkdir(exist_ok=True)

LABELS = ["greeting", "simple", "complex", "ambiguous", "injection"]
ROUTE = {"greeting": "local", "simple": "flash", "ambiguous": "flash",
         "complex": "pro", "injection": "blocked"}
ROUTES = ["local", "flash", "pro", "blocked"]


def normalize(text: str) -> str:
    """Same cleaning as the router's preprocess_text."""
    text = text.lower().strip()
    text = re.sub(r"[?!.,،؟]", "", text)
    return " ".join(text.split())


def main() -> None:
    train = pd.read_csv(BASE_DIR / "train_starter.csv")
    test = pd.read_csv(BASE_DIR / "test.csv")

    pipe = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                        sublinear_tf=True, preprocessor=normalize),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    )
    search = GridSearchCV(
        pipe, {"logisticregression__C": [1, 10, 100]},
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="f1_macro",
    )
    search.fit(train["text"], train["label"])
    model = search.best_estimator_
    print(f"Best C = {search.best_params_['logisticregression__C']}  "
          f"(train CV macro-F1 = {search.best_score_:.3f})")

    preds, times = [], []
    for text in test["text"]:
        t0 = time.perf_counter()
        preds.append(model.predict([text])[0])
        times.append((time.perf_counter() - t0) * 1000)
    test["pred"] = preds
    test["gold_route"] = test["label"].map(ROUTE)
    test["pred_route"] = test["pred"].map(ROUTE)
    test["correct5"] = test["pred"] == test["label"]
    test["correct4"] = test["pred_route"] == test["gold_route"]

    lines = [f"# Baseline: TF-IDF + logistic regression on test.csv ({len(test)} queries)\n"]
    lines.append(f"5-class accuracy: {test['correct5'].mean():.3f}")
    lines.append(f"4-route accuracy (comparable to the router baseline): "
                 f"{test['correct4'].mean():.3f}\n")
    lines.append("## 5-class precision / recall / F1\n```")
    lines.append(classification_report(test["label"], test["pred"], labels=LABELS,
                                       zero_division=0, digits=3))
    lines.append("```\n## 5-class confusion matrix (rows = gold, cols = predicted)\n```")
    lines.append(pd.DataFrame(confusion_matrix(test["label"], test["pred"], labels=LABELS),
                              index=LABELS, columns=LABELS).to_string())
    lines.append("```\n## 4-route precision / recall / F1\n```")
    lines.append(classification_report(test["gold_route"], test["pred_route"],
                                       labels=ROUTES, zero_division=0, digits=3))
    lines.append("```\n## 5-class accuracy by language")
    lines.append(test.groupby("lang")["correct5"].mean().round(3).to_string())
    lines.append("\n## 5-class accuracy on hard cases (hard=1) vs normal (hard=0)")
    lines.append(test.groupby("hard")["correct5"].mean().round(3).to_string())
    lines.append(f"\nMedian predict time: {pd.Series(times).median():.2f} ms (local, no API)")

    report = "\n".join(lines)
    print(report)
    (OUT_DIR / "baseline_tfidf.md").write_text(report, encoding="utf-8")
    test.to_csv(OUT_DIR / "baseline_tfidf_predictions.csv", index=False)


if __name__ == "__main__":
    main()