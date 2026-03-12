# Models

This project includes two different model approaches for sentiment analysis:

- a **Transformer-based model** as the primary solution
- a **FastText model** as a lightweight baseline for comparison

The purpose of this design is to compare a modern deep learning approach with a faster and more lightweight baseline, highlighting the trade-off between performance and efficiency.

---

## Transformer Model

The primary model used in this project is:

```
cardiffnlp/twitter-roberta-base-sentiment-latest
```

This is a Transformer model specialized in sentiment analysis on short, noisy text such as social media posts.

### Why this model

The Transformer approach was selected because it provides:

- strong performance on sentiment classification tasks
- contextual understanding of words and phrases
- robustness on short informal text
- better semantic representation than traditional bag-of-words models

### Strengths

- high classification performance
- strong contextual representation
- effective on social-media style language
- suitable for production-grade NLP tasks

### Limitations

- heavier than classical ML baselines
- slower inference compared to FastText
- requires more computational resources

---

## FastText Baseline

FastText is included as a baseline model trained within the project.

It is useful for comparing a lightweight NLP model against the Transformer solution.

### Why FastText

FastText provides:

- fast training
- efficient inference
- low computational cost
- a simple and interpretable baseline

### Strengths

- lightweight and fast
- easy to train
- efficient on CPU
- useful for comparison and fallback scenarios

### Limitations

- lower expressive power than Transformers
- weaker contextual understanding
- more limited performance on complex language patterns

---

## Model Comparison

The two approaches serve different purposes within the project:

| Aspect | Transformer | FastText |
|--------|-------------|----------|
| Performance | Higher | Lower |
| Training Speed | Slower | Faster |
| Inference Speed | Slower | Faster |
| Resource Usage | Higher | Lower |
| Context Understanding | Strong | Limited |
| Production Simplicity | Medium | High |

The Transformer model is the **primary model** for best predictive performance.

FastText is maintained as a **baseline and lightweight alternative**, useful for comparison and potential low-resource deployment scenarios.

---

## Output Classes

Both models classify sentiment into three categories:

- **Positive**
- **Neutral**
- **Negative**

The final prediction pipeline can expose one or both models depending on the inference configuration.

---

## Training Strategy

The project supports separate training workflows for both models.

### Transformer Training

The Transformer training pipeline includes:

- tokenization
- fine-tuning on labeled sentiment data
- evaluation with classification metrics
- experiment tracking

### FastText Training

The FastText pipeline includes:

- text preprocessing
- label formatting
- supervised FastText training
- evaluation on validation and test data

---

## Why Two Models

Using two model families in the same project provides several benefits:

- comparison between modern deep learning and lightweight NLP approaches
- better understanding of performance vs efficiency trade-offs
- support for different deployment scenarios
- stronger MLOps experimentation workflow

This design reflects a practical engineering mindset:  
the best model is not always the lightest, and the fastest model is not always the most accurate.

---

## Future Improvements

Potential extensions for the modeling layer include:

- additional Transformer architectures
- hyperparameter optimization
- model registry integration
- ensemble strategies
- quantized models for lightweight deployment
