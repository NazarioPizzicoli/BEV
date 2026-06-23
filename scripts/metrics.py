import json
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Prova a caricare sklearn e seaborn, altrimenti usa fallback manuale o avvisa
try:
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

def compute_advanced_metrics(all_preds, all_targets, idx2answer, output_path=None):
    """
    Computes Confusion Matrix, Precision, Recall, and F1-Score.
    """
    if not SKLEARN_AVAILABLE:
        print("\n[WARNING] scikit-learn o seaborn non installati. Impossibile calcolare metriche avanzate.")
        print("Installa le dipendenze con: pip install scikit-learn seaborn")
        return None

    y_pred = np.array(all_preds)
    y_true = np.array(all_targets)
    
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    target_names = [idx2answer.get(i, f"ID_{i}") for i in unique_labels]
    
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    # Plotting
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Multiclass Confusion Matrix')
    
    if output_path:
        plt.savefig(output_path)
        print(f"Confusion Matrix saved to {output_path}")
    
    plt.show()
    return report

def run_full_metrics_eval(model, dataloader, idx2answer, device="cuda", save_dir=None):
    """
    Helper function per eseguire l'eval e calcolare le metriche.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    print(f"Esecuzione valutazione su {len(dataloader.dataset)} campioni...")
    
    with torch.no_grad():
        for features, question_enc, targets in dataloader:
            features = features.to(device)
            question_enc = question_enc.to(device)
            
            outputs = model(features, question_enc)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    save_path = Path(save_dir) / "confusion_matrix.png" if save_dir else None
    return compute_advanced_metrics(all_preds, all_targets, idx2answer, output_path=save_path)


"""
    1 from scripts.metrics import run_full_metrics_eval
    2 import json
    3
    4 # 1. Carichiamo i mapping delle risposte (necessari per le etichette del grafico)
    5 with open(DICT_DIR / "NuScenes_val_questions.json", "r") as f:
    6     data = json.load(f)
    7 _, idx2answer = decode_question(data)
    8
    9 # 2. Inizializziamo il modello (es. quello basato su CNN)
   10 from bev.models.bevqa import get_model
   11 model = get_model("cnn").to(device)
   12
   13 # 3. Carichiamo il tuo miglior checkpoint
   14 checkpoint_path = "../checkpoints/best_model_cnn.pth"
   15 model.load_state_dict(torch.load(checkpoint_path, map_location=device))
   16
   17 # 4. Creiamo il dataloader per la validazione
   18 val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
   19
   20 # 5. Eseguiamo la valutazione avanzata
   21 # Questo genererà il report testuale e il plot della Confusion Matrix
   22 report = run_full_metrics_eval(
   23     model=model, 
   24     dataloader=val_dataloader, 
   25     idx2answer=idx2answer, 
   26     device=device,
   27     save_dir="../outputs" # Opzionale: dove salvare l'immagine della matrice
   28 )

   """