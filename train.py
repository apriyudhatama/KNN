import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from sklearn.neighbors import KNeighborsClassifier


# ======================================================
# 0) PATH & OUTPUT FOLDER
# ======================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset_agen_valorant.xlsx")

OUT_DIR = os.path.join(BASE_DIR, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


# ======================================================
# 1) LOAD & VALIDASI DATA
# ======================================================
def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


def save_basic_reports(df: pd.DataFrame) -> None:
    df.head(10).to_csv(os.path.join(TAB_DIR, "sample_10_rows.csv"), index=False)
    df.isna().sum().to_csv(os.path.join(TAB_DIR, "missing_values.csv"))
    pd.Series({"duplicates": df.duplicated().sum()}).to_csv(
        os.path.join(TAB_DIR, "duplicates.csv")
    )
    df.select_dtypes(include=[np.number]).describe().to_csv(
        os.path.join(TAB_DIR, "descriptive_stats.csv")
    )


# ======================================================
# 2) DISTRIBUSI DATA
# ======================================================
def plot_combined_distribution(df: pd.DataFrame):

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["pick_rate", "win_rate", "kda"]])

    pick_scaled = scaled[:, 0]
    win_scaled = scaled[:, 1]
    kda_scaled = scaled[:, 2]

    plt.figure(figsize=(9, 6))

    plt.hist(pick_scaled, bins=15, alpha=0.5, label="Pick Rate")
    plt.hist(win_scaled, bins=15, alpha=0.5, label="Win Rate")
    plt.hist(kda_scaled, bins=15, alpha=0.5, label="KDA")

    plt.axvline(np.mean(pick_scaled), linestyle="--")
    plt.axvline(np.mean(win_scaled), linestyle="--")
    plt.axvline(np.mean(kda_scaled), linestyle="--")

    plt.xlabel("Nilai (Skala Normalisasi 0–1)")
    plt.ylabel("Frekuensi")
    plt.title("Distribusi Normalisasi Pick Rate, Win Rate, dan KDA")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(FIG_DIR, "distribution_overlay.png"), dpi=300)
    plt.close()


# ======================================================
# 3) ENCODING
# ======================================================
def add_category_codes(df: pd.DataFrame) -> pd.DataFrame:
    df_model = df.copy()
    df_model["map_code"] = df_model["map"].astype("category").cat.codes
    df_model["role_code"] = df_model["role"].astype("category").cat.codes
    df_model["agent_code"] = df_model["agent"].astype("category").cat.codes
    return df_model


# ======================================================
# 4) SPLIT DATA
# ======================================================
def prepare_train_test(
    df_model: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
):

    X = df_model[feature_cols].values
    y = df_model[target_col].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(
        X_scaled,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


# ======================================================
# 5) TUNING KNN
# ======================================================
def tune_knn(X_train, X_test, y_train, y_test):

    k_values = range(1, 26)
    acc_scores = []

    best_k = 1
    best_acc = 0
    best_model = None

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        acc_scores.append(acc)

        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_model = model

    plt.figure()
    plt.plot(k_values, acc_scores)
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("KNN: Accuracy vs k")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "knn_accuracy_vs_k.png"), dpi=300)
    plt.close()

    return best_k, best_acc, best_model


# ======================================================
# 6) EVALUASI KNN (CONFUSION MATRIX 2x2)
# ======================================================
def evaluate_knn(model, X_test, y_test):

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="weighted", zero_division=0
    )

    print("\n=== HASIL EVALUASI KNN ===")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    report = classification_report(y_test, pred, zero_division=0)
    with open(os.path.join(TAB_DIR, "classification_report_knn.txt"), "w") as f:
        f.write(report)

    total_data = len(y_test)
    total_benar = np.sum(pred == y_test)
    total_salah = total_data - total_benar

    persen_benar = (total_benar / total_data) * 100
    persen_salah = (total_salah / total_data) * 100

    plt.figure(figsize=(7, 5))
    bars = plt.bar(
        ["Prediksi Benar", "Prediksi Salah"],
        [total_benar, total_salah]
    )

    for bar, value, percent in zip(
        bars,
        [total_benar, total_salah],
        [persen_benar, persen_salah]
    ):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            f"{value}\n({percent:.2f}%)",
            ha='center',
            va='bottom'
        )

    plt.title("Confusion Matrix 2x2 Model KNN")
    plt.ylabel("Jumlah Data Uji")
    plt.tight_layout()

    plt.savefig(os.path.join(FIG_DIR, "confusion_matrix_knn_2x2.png"), dpi=300)
    plt.close()

    print("\n=== RINGKASAN CONFUSION MATRIX ===")
    print(f"Total Data Uji   : {total_data}")
    print(f"Prediksi Benar   : {total_benar} ({persen_benar:.2f}%)")
    print(f"Prediksi Salah   : {total_salah} ({persen_salah:.2f}%)")


# ======================================================
# 7) SISTEM REKOMENDASI KOMPOSISI TIM (TABEL FINAL)
# ======================================================
def generate_team_recommendation(df: pd.DataFrame):

    print("\n=== GENERATING TEAM RECOMMENDATION TABLE ===")

    scaler = MinMaxScaler()
    features = ["pick_rate", "win_rate", "kda", "round_contribution"]

    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])

    recommendation_results = []

    for map_name in df_scaled["map"].unique():

        df_map = df_scaled[df_scaled["map"] == map_name].copy()

        ideal_profile = df_map[features].max().values

        df_map["distance"] = np.linalg.norm(
            df_map[features].values - ideal_profile,
            axis=1
        )

        selected_agents = []
        roles_needed = ["Duelist", "Controller", "Initiator", "Sentinel"]

        team_comp = {"Map": map_name}

        for role in roles_needed:
            df_role = df_map[df_map["role"] == role]
            if not df_role.empty:
                best_agent = df_role.sort_values("distance").iloc[0]
                team_comp[role] = best_agent["agent"]
                selected_agents.append(best_agent["agent"])

        df_remaining = df_map[~df_map["agent"].isin(selected_agents)]
        if not df_remaining.empty:
            flex_agent = df_remaining.sort_values("distance").iloc[0]
            team_comp["Flex"] = flex_agent["agent"]
        else:
            team_comp["Flex"] = "-"

        recommendation_results.append(team_comp)

    df_recommendation = pd.DataFrame(recommendation_results)

    df_recommendation.to_csv(
        os.path.join(TAB_DIR, "rekomendasi_team_comp_knn.csv"),
        index=False
    )

    print("Rekomendasi disimpan ke outputs/tables/rekomendasi_team_comp_knn.csv")

    return df_recommendation


# ======================================================
# MAIN
# ======================================================
def main():

    df = load_data(DATA_PATH)
    save_basic_reports(df)

    plot_combined_distribution(df)

    df_model = add_category_codes(df)

    feature_cols = [
        "map_code",
        "role_code",
        "pick_rate",
        "win_rate",
        "kda",
        "round_contribution"
    ]

    X_train, X_test, y_train, y_test = prepare_train_test(
        df_model,
        feature_cols,
        "agent_code"
    )

    best_k, best_acc, best_model = tune_knn(
        X_train, X_test, y_train, y_test
    )

    print(f"\nBest k: {best_k} | Accuracy: {best_acc:.4f}")

    evaluate_knn(best_model, X_test, y_test)

    generate_team_recommendation(df)

    print("\nDONE")


if __name__ == "__main__":
    main()
