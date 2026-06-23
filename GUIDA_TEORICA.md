# Guida Teorica: BEV-VQA (Bird's Eye View Visual Question Answering)

## 🎯 Obiettivo del Progetto
L'obiettivo è creare un modello in grado di interpretare scene complesse di guida autonoma rappresentate in formato **BEV (Bird's Eye View)** e rispondere a domande in linguaggio naturale. Questo approccio è fondamentale per l'**explainability** dei sistemi di guida autonoma: non solo l'auto decide cosa fare, ma "dimostra" di comprendere la scena rispondendo a domande su di essa.

---

## 🏛️ L'Architettura del Modello

Il modello segue una filosofia **Encoder-Fusion-Decoder**.

### 1. Vision Encoder: BEV Adapters
Il BEV è un tensore `[128, 200, 200]`. Per processarlo, abbiamo tre varianti:
-   **CNN Adapter**: Usa una serie di layer convoluzionali (`Conv2d`, `ReLU`) per ridurre la risoluzione spaziale da 200x200 a 10x10, aumentando la densità informativa a 512 canali. È ottimo per catturare pattern locali e oggetti ben definiti.
-   **ViT Adapter (Vision Transformer)**: 
    -   Divide la mappa BEV in **patch** da 20x20.
    -   Esegue un **Linear Embedding** di ogni patch.
    -   Aggiunge **Positional Encodings** apprendibili per mantenere l'informazione sulla posizione spaziale.
    -   Processa le patch tramite un **Transformer Encoder** (2 layer, 8 heads). Questo permette al modello di capire relazioni a lunghissima distanza (es. un'auto a 50 metri che influenza una decisione all'incrocio).
-   **Linear Adapter**: Una baseline semplice che usa Adaptive Average Pooling per ridurre drasticamente la complessità.

### 2. Language Encoder: GloVe + Projection
-   Le domande sono tokenizzate e mappate su vettori **GloVe** (300 dimensioni).
-   Vengono portate a una lunghezza fissa (es. 30-35 token) tramite padding.
-   Un **TextAdapter** (layer lineare) proietta i vettori da 300 a 512 dimensioni per renderli compatibili con le feature visive.

### 3. Fusion: Multi-modal Co-Attention (MCA)
Questo è il componente più avanzato. Implementa una **Co-Attention simmetrica**:
1.  **Self-Attention (BEV & Text)**: Ogni modalità "riflette" su se stessa. Il testo capisce le dipendenze grammaticali, il BEV capisce la disposizione spaziale degli oggetti.
2.  **Cross-Attention**: Usiamo le feature del testo come *Query* e le feature del BEV come *Key* e *Value*. 
    -   In pratica: "Per rispondere alla parola 'pedone', guarda in quali zone della mappa BEV ci sono feature simili a un pedone".
    -   Questo crea una mappa di attenzione che evidenzia le regioni visive rilevanti per la specifica domanda.

### 4. Head: Global Pooling & Classification
-   Prendiamo l'output del modulo di attenzione (`[Batch, Seq_Len, 512]`).
-   Eseguiamo un **Global Average Pooling** sulla sequenza per ottenere un singolo vettore di feature multimodali.
-   Un layer lineare finale (`OutputHead`) classifica questo vettore in una delle **30 classi** (risposte predefinite).

---

## 📊 Dataset: NuScenes-QA
Il progetto utilizza una derivazione del dataset **NuScenes**, uno dei più completi per la guida autonoma:
-   **Scene Realistiche**: Dati raccolti a Boston e Singapore.
-   **Multimodalità**: Dati provenienti da 6 camere, Lidar e Radar, fusi nella rappresentazione BEV.
-   **Domande**: Generate sinteticamente o annotate, riguardano la presenza, il colore, la posizione e l'azione degli oggetti (auto, pedoni, ostacoli).

---

## ⚙️ Iperparametri Chiave
-   **Dimensione Modello (`d_model`)**: 512. È la larghezza dello spazio latente in cui visione e linguaggio "parlano".
-   **Heads di Attenzione**: 8. Permette al modello di guardare 8 aspetti diversi della scena contemporaneamente (es. una head guarda il colore, una la distanza, una la direzione).
-   **Batch Size**: 128 (default). Un buon bilanciamento tra stabilità del gradiente e utilizzo della VRAM.
-   **Optimizer**: Adam con Learning Rate 1e-4.
