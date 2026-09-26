# 📊 Sequence & Structure Representation Permutation Benchmarking Report
## Systematic Ablation of ESM-2 Sequence Embeddings and AlphaFold Predicted Structures

This report compiles the zero-shot cross-species transfer performance (trained on HIV-1, tested zero-shot on HIV-2 Susceptibilities) across all **16 representation combinations** of sequence and structural encoders.

---

## 1. Top Performing Representation Configuration

*   **Sequence Representation**: **ESM-2 PCA Only**
*   **Structural Representation**: **Crystal Coordinates**
*   **Best ML Model**: **XGBoost**
*   **Feature Dimension**: **172-D**
*   **Zero-Shot Pearson $R$**: **+0.3371**
*   **Zero-Shot Spearman $\rho$**: **+0.2875**
*   **Zero-Shot RMSE**: **1.7258 kcal/mol**
*   **Zero-Shot p±1.0 Accuracy**: **45.6%**

---

## 2. Complete Representation Ablation Matrix

| Sequence Representation | Structural Representation | Model | Dim | Pearson R | Spearman ρ | RMSE | p1.0 Acc |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Scales Only | No Structure | **SVR** | 153 | -0.2132 | -0.1664 | 1.861 | 43.8% |
| Scales Only | No Structure | **RandomForest** | 153 | +0.0250 | -0.0780 | 1.935 | 41.9% |
| Scales Only | No Structure | **XGBoost** | 153 | +0.1034 | +0.1273 | 1.854 | 41.2% |
| Scales Only | Crystal Coordinates | **SVR** | 161 | +nan | +nan | 1.821 | 43.1% |
| Scales Only | Crystal Coordinates | **RandomForest** | 161 | -0.0144 | -0.1277 | 1.883 | 38.8% |
| Scales Only | Crystal Coordinates | **XGBoost** | 161 | +0.0666 | +0.0007 | 2.247 | 38.8% |
| Scales Only | AlphaFold/ESMFold WT | **SVR** | 161 | +nan | +nan | 1.821 | 43.1% |
| Scales Only | AlphaFold/ESMFold WT | **RandomForest** | 161 | +0.1364 | +0.1497 | 1.898 | 43.1% |
| Scales Only | AlphaFold/ESMFold WT | **XGBoost** | 161 | +0.1163 | +0.1141 | 2.254 | 38.1% |
| Scales Only | ESMFold Dynamic PDBs | **SVR** | 161 | +nan | +nan | 1.821 | 43.1% |
| Scales Only | ESMFold Dynamic PDBs | **RandomForest** | 161 | -0.0144 | -0.1277 | 1.883 | 38.8% |
| Scales Only | ESMFold Dynamic PDBs | **XGBoost** | 161 | +0.0666 | +0.0007 | 2.247 | 38.8% |
| ProtBERT PCA Only | No Structure | **SVR** | 164 | +0.1657 | +0.1945 | 1.821 | 43.1% |
| ProtBERT PCA Only | No Structure | **RandomForest** | 164 | -0.0414 | -0.0198 | 1.932 | 40.6% |
| ProtBERT PCA Only | No Structure | **XGBoost** | 164 | +0.2289 | +0.2507 | 1.862 | 44.4% |
| ProtBERT PCA Only | Crystal Coordinates | **SVR** | 172 | +nan | +nan | 1.821 | 43.1% |
| ProtBERT PCA Only | Crystal Coordinates | **RandomForest** | 172 | +0.0749 | +0.0920 | 1.855 | 45.0% |
| ProtBERT PCA Only | Crystal Coordinates | **XGBoost** | 172 | +0.0405 | -0.0010 | 2.093 | 36.9% |
| ProtBERT PCA Only | AlphaFold/ESMFold WT | **SVR** | 172 | +nan | +nan | 1.821 | 43.1% |
| ProtBERT PCA Only | AlphaFold/ESMFold WT | **RandomForest** | 172 | +0.0932 | +0.1027 | 1.849 | 45.0% |
| ProtBERT PCA Only | AlphaFold/ESMFold WT | **XGBoost** | 172 | +0.0569 | +0.0256 | 2.038 | 37.5% |
| ProtBERT PCA Only | ESMFold Dynamic PDBs | **SVR** | 172 | +nan | +nan | 1.821 | 43.1% |
| ProtBERT PCA Only | ESMFold Dynamic PDBs | **RandomForest** | 172 | +0.0749 | +0.0920 | 1.855 | 45.0% |
| ProtBERT PCA Only | ESMFold Dynamic PDBs | **XGBoost** | 172 | +0.0405 | -0.0010 | 2.093 | 36.9% |
| ESM-2 PCA Only | No Structure | **SVR** | 164 | -0.0708 | -0.0083 | 1.821 | 43.1% |
| ESM-2 PCA Only | No Structure | **RandomForest** | 164 | -0.1001 | -0.0964 | 1.959 | 38.8% |
| ESM-2 PCA Only | No Structure | **XGBoost** | 164 | +0.1977 | +0.1290 | 1.869 | 41.2% |
| ESM-2 PCA Only | Crystal Coordinates | **SVR** | 172 | +nan | +nan | 1.821 | 43.1% |
| ESM-2 PCA Only | Crystal Coordinates | **RandomForest** | 172 | -0.2172 | -0.2102 | 1.937 | 40.0% |
| ESM-2 PCA Only | Crystal Coordinates | **XGBoost** | 172 | +0.3371 | +0.2875 | 1.726 | 45.6% |
| ESM-2 PCA Only | AlphaFold/ESMFold WT | **SVR** | 172 | +nan | +nan | 1.821 | 43.1% |
| ESM-2 PCA Only | AlphaFold/ESMFold WT | **RandomForest** | 172 | -0.2806 | -0.2037 | 1.933 | 38.1% |
| ESM-2 PCA Only | AlphaFold/ESMFold WT | **XGBoost** | 172 | +0.0686 | -0.0009 | 1.831 | 39.4% |
| ESM-2 PCA Only | ESMFold Dynamic PDBs | **SVR** | 172 | +nan | +nan | 1.821 | 43.1% |
| ESM-2 PCA Only | ESMFold Dynamic PDBs | **RandomForest** | 172 | -0.2172 | -0.2102 | 1.937 | 40.0% |
| ESM-2 PCA Only | ESMFold Dynamic PDBs | **XGBoost** | 172 | +0.3371 | +0.2875 | 1.726 | 45.6% |
| ProtBERT + ESM-2 PCA | No Structure | **SVR** | 196 | -0.0652 | +0.1326 | 1.821 | 43.1% |
| ProtBERT + ESM-2 PCA | No Structure | **RandomForest** | 196 | -0.0751 | -0.0326 | 2.038 | 39.4% |
| ProtBERT + ESM-2 PCA | No Structure | **XGBoost** | 196 | +0.1095 | +0.0464 | 1.875 | 41.9% |
| ProtBERT + ESM-2 PCA | Crystal Coordinates | **SVR** | 204 | +nan | +nan | 1.821 | 43.1% |
| ProtBERT + ESM-2 PCA | Crystal Coordinates | **RandomForest** | 204 | -0.1233 | -0.1038 | 2.123 | 39.4% |
| ProtBERT + ESM-2 PCA | Crystal Coordinates | **XGBoost** | 204 | -0.0956 | -0.1076 | 1.974 | 36.9% |
| ProtBERT + ESM-2 PCA | AlphaFold/ESMFold WT | **SVR** | 204 | +nan | +nan | 1.821 | 43.1% |
| ProtBERT + ESM-2 PCA | AlphaFold/ESMFold WT | **RandomForest** | 204 | -0.1120 | -0.0820 | 2.117 | 41.2% |
| ProtBERT + ESM-2 PCA | AlphaFold/ESMFold WT | **XGBoost** | 204 | -0.2420 | -0.2464 | 2.053 | 35.6% |
| ProtBERT + ESM-2 PCA | ESMFold Dynamic PDBs | **SVR** | 204 | +nan | +nan | 1.821 | 43.1% |
| ProtBERT + ESM-2 PCA | ESMFold Dynamic PDBs | **RandomForest** | 204 | -0.1233 | -0.1038 | 2.123 | 39.4% |
| ProtBERT + ESM-2 PCA | ESMFold Dynamic PDBs | **XGBoost** | 204 | -0.0956 | -0.1076 | 1.974 | 36.9% |
| Scales + ProtBERT + ESM-2 | No Structure | **SVR** | 217 | +0.0664 | +0.0185 | 1.821 | 43.1% |
| Scales + ProtBERT + ESM-2 | No Structure | **RandomForest** | 217 | -0.0216 | -0.0031 | 1.941 | 40.0% |
| Scales + ProtBERT + ESM-2 | No Structure | **XGBoost** | 217 | +0.1566 | +0.2061 | 1.821 | 41.9% |
| Scales + ProtBERT + ESM-2 | Crystal Coordinates | **SVR** | 225 | +nan | +nan | 1.821 | 43.1% |
| Scales + ProtBERT + ESM-2 | Crystal Coordinates | **RandomForest** | 225 | -0.0614 | -0.0274 | 2.006 | 39.4% |
| Scales + ProtBERT + ESM-2 | Crystal Coordinates | **XGBoost** | 225 | +0.0064 | +0.0012 | 1.887 | 38.1% |
| Scales + ProtBERT + ESM-2 | AlphaFold/ESMFold WT | **SVR** | 225 | +nan | +nan | 1.821 | 43.1% |
| Scales + ProtBERT + ESM-2 | AlphaFold/ESMFold WT | **RandomForest** | 225 | -0.0451 | -0.0204 | 2.009 | 40.0% |
| Scales + ProtBERT + ESM-2 | AlphaFold/ESMFold WT | **XGBoost** | 225 | +0.0349 | +0.0868 | 1.895 | 40.0% |
| Scales + ProtBERT + ESM-2 | ESMFold Dynamic PDBs | **SVR** | 225 | +nan | +nan | 1.821 | 43.1% |
| Scales + ProtBERT + ESM-2 | ESMFold Dynamic PDBs | **RandomForest** | 225 | -0.0614 | -0.0274 | 2.006 | 39.4% |
| Scales + ProtBERT + ESM-2 | ESMFold Dynamic PDBs | **XGBoost** | 225 | +0.0064 | +0.0012 | 1.887 | 38.1% |

---

## 3. Key Findings & Biological Insights

### 3.1 ESM-2 Sequence Representation Impact
*   **ProtBERT vs. ESM-2**: Compare the performance of the lightweight 320-D ESM-2 sequence embeddings (PCA-reduced to 32-D) against ProtBERT's 1024-D (PCA-reduced to 32-D) representations. ESM-2 sequence-level average features capture structural evolutionary constraints from ESMFold training, yielding a highly focused representation.
*   **Combined Embeddings (ProtBERT + ESM-2)**: Concatenating both language model PCAs (64-D sequence space) expands the evolutionary context, potentially resolving strain-specific mutations that single encoders miss.

### 3.2 AlphaFold WT vs. Crystal vs. ESMFold coordinates
*   **Crystal Template**: Serves as the static experimental baseline (using high-resolution structures `1IDA`/`3S45`).
*   **AlphaFold/ESMFold WT Dimer**: Evaluates the impact of replacing experimental crystal structures with a computationally predicted WT fold. 
*   **ESMFold Dynamic PDBs**: Encodes sequence-specific 3D fluctuations and distances calculated on-the-fly for each distinct mutant complex, capturing localized side-chain movements.

### 3.3 Recommendation on Integration
*   If **ESM-2** or **AlphaFold WT Dimer** improves the zero-shot Pearson correlation $R$ by $\ge 0.01$ over the locked crystal baseline ($R = 0.39$ on Biophysics-only, or the overall RF sequence baseline), they should be locked as the new feature representation standard.

---
*Report generated by run_representation_ablation.py | Date: 2026-06-16*
