## Entity Resolution with Knowledge Graphs

This project implements a complete **Entity Resolution (ER)** pipeline for matching and clustering organization affiliation strings (such as universities, companies, and research institutions) using both traditional machine learning and graph-based methods.

### Overview
The pipeline follows a structured process consisting of several key stages:

1. **Data Normalization & Preprocessing**  
   A **Named Entity Recognition (NER)** model is applied to extract relevant entities (organizations, locations, persons) from affiliation strings.  
   Extracted entities are then normalized through acronym expansion, stopword removal, and country name standardization to ensure consistent textual representation.

2. **Blocking**  
   TF-IDF vectorization combined with k-Nearest Neighbors (kNN) to efficiently reduce candidate pairs and improve scalability.

3. **Feature Extraction**  
   Computation of string, cosine and phonetic similarity metrics, including Edit Ratio, Jaro–Winkler, Longest Common Subsequence, Jaccard, Cosine, TF-IDF, and d_metaphone.

4. **Pairwise Classification**  
   Supervised models (Logistic Regression, Random Forest, XGBoost) trained with stratified cross-validation and Fbeta-optimized threshold tuning to classify candidate pairs as matches or non-matches.

5. **Constraint Filtering & Clustering**  
   Application of geographic and logical constraints followed by transitive closure via **Disjoint Set Union (DSU)** to merge matched entities into consistent clusters.

6. **Evaluation**  
   Assessment using pairwise metrics to evaluate predicted clusters to gold-standard mappings.

### Dataset
The dataset used for training and evaluation was derived from the **Benchmark Datasets for Entity Resolution** provided by the University of Leipzig Database Group:  
 - [https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution)

### References
- Leichen Zhang, *Entity Resolution Lecture Notes – HKUST MSCIT6000D*  
  [https://home.cse.ust.hk/~leichen/courses/mscit6000d/notes/entityresolution.pdf](https://home.cse.ust.hk/~leichen/courses/mscit6000d/notes/entityresolution.pdf)  
- Spot Intelligence (2024), *Entity Resolution: Understanding the Process*  
  [https://spotintelligence.com/2024/01/22/entity-resolution/](https://spotintelligence.com/2024/01/22/entity-resolution/)

### Outcome
The resulting system produces high-precision entity clusters suitable for integration into **Knowledge Graphs**, enabling consistent and de-duplicated representation of real-world entities across noisy textual datasets.
