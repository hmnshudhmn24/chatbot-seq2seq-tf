# 💬 Chatbot using TensorFlow and Seq2Seq

This project demonstrates how to build a simple chatbot using an encoder-decoder architecture (Seq2Seq) with attention mechanism in TensorFlow.

## 🔍 Overview

- Uses a small set of question-answer pairs
- Implements Seq2Seq with LSTM and Bahdanau attention
- Trains a chatbot to understand and respond to basic prompts

## 📦 Dependencies

```bash
pip install tensorflow numpy matplotlib
```

## 🚀 How to Run

1. Install the dependencies listed above.
2. Run the script:
```bash
python chatbot_seq2seq.py
```

## 📁 Files

- `chatbot_seq2seq.py`: Main Python script
- `README.md`: Documentation

## 💡 Example

You can modify the `questions` and `answers` lists in the script to train on different conversation pairs.

## 📚 Reference

- TensorFlow official Seq2Seq tutorials
- Chatbot concepts with attention mechanism