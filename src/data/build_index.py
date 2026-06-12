import json
from pathlib import Path
from typing import List, Union


def build_filtered_dataset(
    features_dir: Union[str, Path],
    json_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
):
    features_dir = Path(features_dir)

    # 1. Crea un set con tutti i sample_token validi (senza l'estensione .pt)
    # Esempio: "0a3abc33048d46f9bd78151d1df4b004.pt" -> "0a3abc33048d46f9bd78151d1df4b004"
    print("Scansionando la cartella delle features...")
    valid_tokens = {file_path.stem for file_path in features_dir.glob("*.pt")}
    print(f"Trovate {len(valid_tokens)} BEV features uniche.")

    filtered_questions = []

    # 2. Carica e filtra le domande dai file JSON
    for json_path in json_paths:
        print(f"Elaborazione di {json_path}...")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = data.get("questions", [])
        initial_count = len(questions)

        # Filtra mantenendo solo le domande il cui sample_token ha una feature corrispondente
        kept_questions = [q for q in questions if q["sample_token"] in valid_tokens]
        filtered_questions.extend(kept_questions)

        print(
            f"  -> Domande originali: {initial_count} | Domande mantenute: {len(kept_questions)}"
        )

    # 3. Crea la nuova struttura del dizionario
    new_dataset = {
        "info": {
            "split": "train_filtered",
            "version": "1.0",
            "description": "Filtered dataset containing only questions with available BEV features in folder.",
        },
        "questions": filtered_questions,
    }

    # 4. Salva il nuovo JSON
    print(f"Salvataggio del nuovo dataset in {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_dataset, f, indent=4)

    print(f"Fatto! Totale domande finali: {len(filtered_questions)}")


# ==========================================
# Esecuzione
# ==========================================
if __name__ == "__main__":
    # FEATURES_FOLDER = "/home/robesafe-sandra/BEV/data/dataset_mini/bev_features_mini/train_mini/"
    FEATURES_FOLDER = (
        "/home/robesafe-sandra/BEV/data/dataset_mini/bev_features_mini/val_mini/"
    )
    JSON_FILES = [
        "/home/robesafe-sandra/BEV/data/dataset_mini/dict/NuScenes_train_questions.json",
        "/home/robesafe-sandra/BEV/data/dataset_mini/dict/NuScenes_val_questions.json",
    ]
    OUTPUT_FILE = "/home/robesafe-sandra/BEV/data/dataset_mini/dict/NuScenes_val_questions_mini.json"

    build_filtered_dataset(FEATURES_FOLDER, JSON_FILES, OUTPUT_FILE)
