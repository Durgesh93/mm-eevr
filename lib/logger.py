from pathlib import Path 
import numpy as np
import pandas as pd
# ============================================================
# Utility: simple logger
# ============================================================
class Logger:
    def __init__(self, model_name, modality, task, experiment, threshold=0.5):
        self.file_path = Path("output") / f"{model_name}-{modality}-{task}-{experiment}-results.txt"
        self.npz_path  = Path("output") / f"{model_name}-{modality}-{task}-{experiment}-predictions.npz"

        self.threshold = threshold
        self.pred_class_col = f"pred_class_{self.threshold}"

        self.prediction_rows = []

        self.file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write("")
    
    def print(self, message=""):
        message = str(message)
        print(message)

        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def add_predictions(
        self,
        seed,
        fold,
        pid,
        sample_id,
        y_true,
        pred_proba,
    ):
        y_true = np.asarray(y_true).astype("int32")
        pred_proba = np.asarray(pred_proba).astype("float32")

        pred_class = (
            pred_proba >= self.threshold
        ).astype("int32")

        self.prediction_rows.append(
            pd.DataFrame({
                "seed": np.repeat(seed, len(y_true)),
                "fold": np.repeat(fold, len(y_true)),
                "pid": np.repeat(str(pid), len(y_true)),
                "sample_id": np.asarray(sample_id).astype(str),
                "gt": y_true,
                "pred_proba": pred_proba,
                self.pred_class_col: pred_class,
            })
        )

    def save_npz(self):
        if len(self.prediction_rows) == 0:
            self.print("No prediction probabilities to save.")
            return None

        pred_df = pd.concat(
            self.prediction_rows,
            ignore_index=True,
        )

        save_dict = {
            "seed": pred_df["seed"].to_numpy(),
            "fold": pred_df["fold"].to_numpy(),
            "pid": pred_df["pid"].to_numpy(),
            "sample_id": pred_df["sample_id"].to_numpy(),
            "gt": pred_df["gt"].to_numpy(),
            "pred_proba": pred_df["pred_proba"].to_numpy(),
            self.pred_class_col: pred_df[self.pred_class_col].to_numpy(),
            "threshold": np.asarray(self.threshold),
        }

        np.savez_compressed(
            self.npz_path,
            **save_dict,
        )

        self.print(f"Saved LOSO predictions to: {self.npz_path}")

        return self.npz_path
