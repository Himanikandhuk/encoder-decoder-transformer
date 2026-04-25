# 🚀 Transformer from Scratch for NL → Code Generation

## 📌 Overview
This project implements a **Transformer-based encoder–decoder architecture from scratch in PyTorch** for **natural language into executable program code**(includes the SQL and Python tokensizer).
### The Implemented model architecture
<img width="453" height="680" alt="llm_flow_white-Photoroom (1)" src="https://github.com/user-attachments/assets/eb7d6790-41df-4f07-929c-497fb93823bd" />

Unlike pre-trained models, this implementation focuses on:<br>
- Full transparency of architecture  <br>
- Modular design for experimentation <br> 
- Understanding performance under computational constraints  <br>

---

## 🎯 Key Features<br>
- Complete Transformer implementation (Encoder + Decoder)<br>
- Built from scratch (no pre-trained models)<br>
- Supports SQL and Python code generation<br>
- Custom tokenization (with structural tokens like `<IND>`, `<NL>`)<br>
- Modular and configurable architecture<br>
- GPU-compatible training (Colab / CUDA)<br>

---

## Core Components:<br>
- Multi-Head Attention  <br>
- Scaled Dot-Product Attention  <br>
- Positional Encoding (Sinusoidal)  <br>
- Feed-Forward Networks  <br>
- Residual Connections + Layer Normalization  <br>

---

## ⚙️ Configurable Parameters<br>
The model is designed to be fully configurable:<br><br>

- `d_model` → Embedding dimension  <br>
- `num_heads` → Number of attention heads  <br>
- `num_layers` → Encoder/Decoder depth  <br>
- `d_ff` → Feed-forward dimension  <br>
- `dropout` → Regularization  <br>
- `seq_len` → Maximum sequence length  <br>
- `batch_size` → Training batch size  <br>
- `learning_rate` → Optimizer learning rate<br>
Modify these in **training_script.py**<br>

---

## 🧠 Architecture Summary<br>
Although these parameters are variable ,  the current **training_script.py** has the following parameters initialized:<br><br>
- Encoder Layers: 6  <br>
- Decoder Layers: 6  <br>
- Total Attention Heads: 144 <br> 
- Model Dimension: Configurable (default ~384–412)<br>

---

## 📂 Project Structure<br>
├── attention_core.py<br>
├── encoder_stack.py<br>
├── decoder_stack.py<br>
├── input_modules.py<br>
├── transformer_model.py<br>
├── python_tokenizer_util.py<br>
├── sql_tokenizer_util.py<br>
├── training_script.py<br>
├── inference_util.py<br>
├── requirements.txt<br>
└── README.md<br>

---

## Model Pipeline<br>
Input → Tokenizer → Embedding → Positional Encoding
→ Encoder → Decoder → Linear → Softmax → Output Code<br>

---

## ▶️ Usage<br><br>

### Train<br>
python training_script.py<br>
### Inference<br>
python inference_util.py<br>

---

## Usage
This architecture can be used on any other usecase also , with its corresponding tokenizer_util.py file,training_script.py and inference_util.py

---

## Post-training results
The results after training the model on the usecases are:
<img width="898" height="419" alt="image" src="https://github.com/user-attachments/assets/5ef53bcf-4ed0-4928-9ea2-0cb06603fc3b" />



## 📌 Under the Guidance of 

- Dr. O. B. V. Ramanaiah
- Senior Professor
- JNTUH 
