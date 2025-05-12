# CodyDroid: Ultra-Lightweight Code Generation Models for Native Android

Code generation is an increasingly popular domain, with tools emerging to help developers write code more efficiently and learn better practices. While most tools focus on general or language-specific code generation, **CodyDroid** is specifically designed to serve **Native Android Developers** by providing high-quality, Android SDK-specific code snippets in Kotlin and Java.

---

## 🔍 Overview

**CodyDroid** introduces a family of **ultra-lightweight LLMs** trained to generate Native Android code. The project is motivated by the vibrant Android developer community and aims to:
- Reduce repetitive coding effort.
- Assist beginners in learning Android development best practices.
- Provide a lightweight alternative to massive code generation models.

---

## 🧠 Models

Two distinct models have been trained and evaluated:

| Model Name                     | Base Language | Accuracy | Size     |
|-------------------------------|---------------|----------|----------|
| `deepseek-coder-5.9M-kexer`   | Kotlin        | 59.99%   | 5.9M     |
| `codegen-175K-mono-java`      | Java          | 65.84%   | 175K     |

### 🎯 Evaluation Criteria

Each model was evaluated on 4 key coding dimensions:

| Concept                  | Model-1 (Kotlin) | Model-2 (Java) |
|--------------------------|------------------|----------------|
| Logical Implementation   | 70%              | 80%            |
| Object-Oriented Design   | 50%              | 70%            |
| Library/API Understanding| 80%              | 60%            |
| Performance Optimization | 0%               | 20%            |

> **Note**: Model-2 (Java) slightly outperforms Model-1 overall but tends to follow traditional patterns, while Model-1 reflects modern development practices.

---

## 📁 Project Structure

```
CodyDroid/
├── gemini_results/ # Comparative results with Gemini 
│ ├── code_comparison_gemini.csv
│ ├── code_generation_gemini.csv
│ ├── model_performance_gemini.txt
├── results/ # Performance and comparison results
│ ├── model_performance_summary_Model-1.txt
│ ├── model_performance_summary_Model-2.txt
│ ├── code_comparison_with_differences_Model-1.csv
│ ├── code_comparison_with_differences_Model-2.csv
│ ├── performance_comparison_with_gemini.ipynb
├── assets/ # Datasets used
│ ├── fine-tuning.csv
│ ├── evaluation.csv
│ ├── fine-tuning-small.csv
│ ├── evaluation-small.csv
│ ├── Dataset_Creation.ipynb
├── src/ # Core training and evaluation code
│ ├── ModelRunner.py
│ ├── ModelTrainer.py
│ ├── run.py
│ ├── train.py
├── requirements.txt # Python dependencies
```

## 🧪 Datasets

Two custom datasets were curated:

- `fine-tuning.csv`: Used for model training.
- `evaluation.csv`: Used for performance benchmarking.

Both datasets cover multiple Android components categorized into:
- **Architecture**
- **Frameworks**
- **Libraries**
