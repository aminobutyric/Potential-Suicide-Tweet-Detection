# Potential Suicide Tweet Detection

## Overview

In this study, we analyze a dataset using multiple machine learning techniques and models to detect potential suicide-related tweets. The primary dataset, referred to as 'data', undergoes preprocessing, while its copy, 'data_zero_shot', is reserved for zero-shot classification utilizing the Hugging Face pipeline (MoritzLaurer/bge-m3-zeroshot-v2.0).

## Techniques and Models

### Classical Bag of Words and TfIdf

We employ the following algorithms for text classification:

- **Naive Bayes**: A probabilistic classifier based on applying Bayes' theorem.
- **SGD Classifier**: A linear classifier optimized using stochastic gradient descent.
- **Logistic Regression**: A statistical model that in its basic form uses a logistic function to model a binary dependent variable.

### Pretrained GloVe Word Embeddings

To enhance our models with semantic understanding, we use GloVe (Global Vectors for Word Representation) embeddings. We employ word-averaging to generate tweet encodings from word embeddings and use these embeddings as input features for our classifiers:

- **Logistic Regression**: Utilized here as the primary classification algorithm to evaluate the effectiveness of word embeddings.

### Zero-shot Classification

For the zero-shot classification tasks, we employ two different approaches:

- **Using a Pretrained NLI Model**: We leverage a pretrained Natural Language Inference (NLI) model capable of classifying texts into categories not seen during training, offering a flexible approach to text classification without the need for extensive labeled data.

- **Using microsoft/Phi-3-mini-4k-instruc LLM**: At the end of the notebook, we utilize the microsoft/Phi-3-mini-4k-instruc large language model (LLM) for zero-shot classification by providing it with specific instructions. This model allows us to extend our classification capabilities further, demonstrating the power of instruction-tuned LLMs in practical applications.

## Summary of Results

### Classical Bag of Words and TfIdf Models

- **Naive Bayes, SGD Classifier, Logistic Regression**: These models performed reasonably well on our dataset, demonstrating good accuracy. However, they lacked generalization and were not suitable for practical applications due to their dependence on the specific training data.

### Pretrained GloVe Word Embeddings

- When using word-averaging for tweet encoding, the performance was similar to the classical models, but with even lower scores. This indicates that the GloVe embeddings did not significantly enhance the classification accuracy in our context.

### Zero-shot Classification using Pretrained NLI Model

- This approach yielded the best results, achieving an accuracy of 87% on unseen data. The zero-shot capability of the NLI model provided robust performance without the need for extensive labeled training data, making it a highly effective method for our task.

### Zero-shot Classification using microsoft/Phi-3-mini-4k-instruc LLM

- Despite the potential of instruction-tuned LLMs, this model did not perform well with our specific instructions, resulting in a score of 50%. This suggests that the model may require further tuning or different instructional approaches to achieve better performance. For enhancement purposes, you can modify the instruction of the LLM and analyze the performance.

## Conclusion

In this notebook, we explored and compared various machine learning techniques for text classification, including classical models, pretrained word embeddings, and advanced zero-shot learning methods.

The comparison of these techniques highlights the strengths and limitations of each approach. Classical models like Bag of Words and TfIdf, along with GloVe embeddings, showed limited generalization capabilities. The zero-shot NLI model demonstrated superior performance, making it the most effective method for our text classification task. Conversely, the instruction-tuned LLM did not meet expectations, indicating the need for further research and refinement in leveraging such models for practical applications.

Future work may involve experimenting with other zero-shot models, fine-tuning LLMs, and exploring hybrid approaches to improve classification accuracy and generalization.

---

Feel free to reach out for any questions or suggestions regarding this project!
