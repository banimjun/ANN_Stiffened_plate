# Artificial Neural Network Model to Assist in Design of Ship Stiffened Plates Considering Ultimate Strength

This repository provides the trained Artificial Neural Network (ANN) model developed for the paper:

**_“Artificial Neural Network Model to Assist in Design of Ship Stiffened Plates Considering Ultimate Strength”_**

The ANN model is designed to assist structural designers in the early-stage design of ship stiffened plates by predicting plate-related parameters based on ultimate strength characteristics.

---

## Contents

- `__main__.py`, `model.py`, `dataset.py`: Python scripts used to train and test the model.
- `allcombine.csv`: Input dataset used for model training.
- `saved_model_tp.h5`: Trained ANN model saved in HDF5 format.
- `requirements.txt`: Required Python packages for training and inference.

---

## Installation & Setup

We recommend using **[PyCharm](https://www.jetbrains.com/pycharm/)** or any Python environment that supports virtual environments.

### 1. Clone the repository

```bash
git clone https://github.com/your-repo-url.git
cd your-repo-folder
```

### 2. Install required packages

```bash
pip install -r requirements.txt
```

### 3. Run the model

```bash
python __main__.py --data_dir=allcombine.csv --method=std
```

You can customize batch size, epochs, and feature columns by modifying `__main__.py`.

---

## Notes

- The model is saved in `.h5` format, which includes the network architecture, weights, and optimizer configuration.
- Use `tensorflow.keras.models.load_model('saved_model_tp.h5')` to reload the trained model.
- The training and evaluation results (e.g., Excel outputs, plots) will be automatically saved.
- The model supports both standardization and min-max normalization as preprocessing options.

---

## Contact

For questions, feedback, or collaboration inquiries, please contact:  
 **june3373@naver.com**

---

## Acknowledgements

Thank you for your interest in this work.  
We hope this ANN model helps improve and accelerate the design process for stiffened plates in marine structures.
