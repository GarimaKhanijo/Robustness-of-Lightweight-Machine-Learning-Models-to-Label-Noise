## **Evaluating the Robustness of Lightweight Machine Learning Models to Label Noise**

### **Description**
This project focuses on understanding how classical machine learning models behave when the training data contains incorrect labels (label noise). In real-world datasets, labels are not always perfect due to human errors, ambiguity, or automated processes, so it becomes important to check how models handle such situations.

In this work, we tested multiple machine learning models under controlled noisy conditions. The idea was not just to check accuracy, but to see how performance changes as noise increases, and which models are more stable compared to others.

### **Features**
- **Evaluation of 6 classical ML models:**

    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Support Vector Machine (SVM)
    - K-Nearest Neighbors (KNN)
    - LightGBM

- **Two types of label noise:**

    - Symmetric noise (random label flipping)
    - Asymmetric noise (directional label flipping)

- **Noise levels tested:**
    0%, 10%, 20%, 30%

- **5-fold stratified cross-validation** used for reliable evaluation

- **Performance metrics:**

    - Accuracy

    - F1-score

- **Additional analysis:**

    - Relative degradation (performance drop due to noise)

    - Stability using standard deviation across folds

    - Comparison of models across datasets

- **Visualization:**

    - Degradation curves

    - Relative drop graphs

    - Heatmaps and ranking plots

### **Project Structure**

├── data/
│   ├── Data.md              # dataset links and details
│   ├── (datasets included)
│
├── notebooks/
│   ├── clean_baseline.ipynb
│   ├── noise_engine.ipynb
│   ├── cross_validation.ipynb
│   └── analysis_plots.ipynb
│
├── results/
│   ├── tables/
│   └── plots/
│
├── literature/
│   └── research_paper.pdf
│
├── requirements.txt
├── README.md


### **Installation**
Make sure Python is installed (recommended: Python 3.8+)

Install dependencies using:
```python
pip install -r requirements.txt
```

### **How to Run the Project**
For best understanding and execution, follow the notebooks (/Notebooks) in order (for both Datasets):

Run `clean_baseline.ipynb`
→ trains models on clean data

Run `noise_engine.ipynb`
→ applies symmetric and asymmetric noise

Run `cross_validation.ipynb`
→ performs 5-fold evaluation

Run `analysis_plots.ipynb`
→ generates graphs and final comparisons

### **Recommended Platforms**
- Jupyter Notebook / JupyterLab
- Google Colab (if running online)
- VS Code (with Python extension)

### **Datasets**
Two datasets are used in this project:
- Breast Cancer Wisconsin (Diagnostic)
- Adult Income Dataset

Dataset source links are available in:
```console
data/Data.md
```

Downloaded datasets are also already included inside the data/ folder for convenience.

### **Results (Overview)**
- All models show performance degradation as noise increases
- **Symmetric noise** causes more damage compared to asymmetric noise
- **SVM** remains the most stable model in most cases
- **Decision Tree** is the most sensitive to noisy labels
- **F1-score** captures performance drop more clearly than accuracy, especially for imbalanced data

Overall, the results show that good performance on clean data does not always mean the model is robust in real-world noisy conditions.

### **Literature**

The complete research paper for this project is available in:

```console
literature/
```
This includes detailed methodology, experiments, and discussion.

### **Authors**
- Garima
- Gurnoor Dhaliwal
- Gagan Deep
- Divesh Gupta

