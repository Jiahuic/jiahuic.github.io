---
layout: page
permalink: /teaching/math-499v599v/lec1/
title: Introduction to Machine Learning
---

## Artificial Intelligence vs Machine Learning vs Deep Learning
* Artificial Intelligence (AI): The science of making machines smart that can mimic human behavior.
* Machine Learning (ML): A subset of AI that uses statistical methods to enable machines to improve with experience.
* Deep Learning (DL): A subset of ML that uses neural networks to enable machines to improve with experience. Most of the recent advances in AI are due to DL and development of faster GPUs.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/math-499v599v/lec1_ML_DP.png" title="ML_DP" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

## Fundamental Concepts of Machine Learning
The learning process of developing a traditional method (e.g., a computer program or a numerical model) is flipped by the machine learning process. In the traditional method, we start with a set of rules and data, and then we develop a computer program or a numerical model. In the machine learning process, we start with a computer program or a numerical model and data, and then we develop a set of rules. The machine learning process is also called the data-driven process.

For example, if you want to make your favorite cheeseburger, you can either follow a recipe or develop your own recipe. In the traditional method, you start with a recipe and then you make your cheeseburger. In the machine learning process, you start with eating enough cheeseburgers and then you develop a recipe. 
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/math-499v599v/lec1_burger.png" title="burger" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/math-499v599v/lec1_burger_ingredient.png" title="burger_ingredient" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

## What ML/DL is good for?
The development of ML/DL is rapidly evolving and the applications are growing exponentially. The pros and cons of ML/DL are discussed below but not restricted to the following.
* **Tackling complex problem with elusive rules**: ML and DL excel in scenarios where defining explicit rules is challenging or impractical. These technologies are adept at uncovering underlying patterns in complex datasets, making them ideal for solving problems that are too intricate for traditional rule-based approaches.

* **Adapting to Continuously Changing Environments**: These methods are well-suited for dynamic environments where conditions and variables are constantly evolving. ML/DL can continuously learn and adapt from new data, making them invaluable in contexts where flexibility and adaptability are key.

* **Analysing Large Data Collections**: ML/DL thrive on large datasets. They can efficiently process and extract meaningful insights from extensive collections of data, a task that would be overwhelming and time-consuming for humans. This capability makes them essential tools for data-rich domains.

## What ML/DL is not good for?
1. **Requirement for Explainability**: ML/DL models, especially deep learning models, are often referred to as "black boxes" due to their complex and non-transparent nature. In contexts where it's crucial to understand and explain how decisions are made (such as legal or medical decision-making), these models may not be ideal.

2. **Superior Performance of Traditional Methods**: In some cases, especially where problems are simple or well-understood, traditional algorithmic approaches may outperform ML/DL models in terms of efficiency, accuracy, or simplicity.

3. **Low Tolerance for Errors**: ML/DL models inherently have a margin of error, as they learn from data and make probabilistic predictions. In high-stakes environments where errors can have significant consequences (like life-critical medical systems or safety-critical automotive systems), these models might not be the best choice.

4. **Scarcity of Data**: The performance of ML/DL models heavily depends on the quantity and quality of the available data. In scenarios where data is scarce, incomplete, or of poor quality, these models may fail to learn effectively and produce reliable results.

## Machine Learning vs Deep Learning
| Machine Learning | Deep Learning |
| :---: | :---: |
| Effective with smaller datasets | Requires large datasets |
| Can often be run on low-end systems | Require more powerfull hardware |
| Interpretable | Less interpretable |
| Structured data | Unstructured data |

### Structured vs Unstructured Data
**Structured Data**: This type of data is highly organized and formatted in a way that makes it easily searchable and identifiable in databases. Structured data follows a specific schema or model, such as rows and columns in a spreadsheet or relational database. Examples include names, dates, addresses, credit card numbers, and stock information.

**Unstructured Data**: Unstructured data lacks a predefined format or organization, making it more complex to process and analyze using conventional database techniques. It comes in various forms, such as text, images, videos, emails, social media posts, and web pages. This data is typically text-heavy, but it may also contain dates, numbers, and facts.

## Types of Machine Learning
* **Supervised Learning**: Supervised learning is a type of machine learning algorithm that uses a known dataset (called the training dataset) to make predictions. The training dataset includes input data and response values. The input data is used to train the model, while the response values tell the model what to do with the input data. The goal of supervised learning is to train the model to correctly identify the response value for new data. Supervised learning is commonly used in applications where historical data predicts likely future events. For example, it can anticipate when credit card transactions are likely to be fraudulent or which insurance customer is likely to file a claim.
* **Unsupervised Learning**: Unsupervised learning is a type of machine learning algorithm used to draw inferences from datasets consisting of input data without labeled responses. The most common unsupervised learning method is cluster analysis, which is used for exploratory data analysis to find hidden patterns or grouping in data. The clusters are modeled using a measure of similarity which is defined upon metrics such as Euclidean or probabilistic distance.
* **Reinforcement Learning**: Reinforcement learning is a type of machine learning algorithm that allows the agent to decide the best next action based on its current state, by learning behaviors that will maximize the reward. The agent learns to achieve a goal in an uncertain, potentially complex environment. In reinforcement learning, an artificial intelligence faces a game-like situation. The computer employs trial and error to come up with a solution to the problem. To get the machine to do what the programmer wants, the artificial intelligence gets either rewards or penalties for the actions it performs. Its goal is to maximize the total reward.
* **Transfer Learning**: Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task. It is a popular approach in deep learning where pre-trained models are used as the starting point on computer vision and natural language processing tasks given the vast compute and time resources required to develop neural network models on these problems and from the huge jumps in skill that they provide on related problems. Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task. It is a popular approach in deep learning where pre-trained models are used as the starting point on computer vision and natural language processing tasks given the vast compute and time resources required to develop neural network models on these problems and from the huge jumps in skill that they provide on related problems.
* **Federal Learning**: Federated learning is a machine learning technique that trains an algorithm across multiple decentralized edge devices or servers holding local data samples, without exchanging them. This approach stands in contrast to traditional centralized machine learning techniques where all the local datasets are uploaded to one server, as well as to more classical decentralized approaches which often assume that local datasets are identically distributed. Federated learning enables multiple actors to build a common, robust machine learning model without sharing data, thus addressing issues of data privacy, data security, data access time, communication bandwidth, and energy efficiency.

## Types of Learning Tasks
1. **Classification**: This involves categorizing data into predefined classes or groups. It's used in applications like spam detection, image recognition, and sentiment analysis.

2. **Regression**: Regression tasks involve predicting a continuous output variable based on one or more input features. Common uses include stock price prediction, real estate valuation, and weather forecasting.

3. **Clustering**: This is about grouping a set of objects in such a way that objects in the same group (cluster) are more similar to each other than to those in other groups. It’s widely used in market segmentation, social network analysis, and search result grouping.

4. **Dimensionality Reduction**: This involves reducing the number of input variables in a dataset, used to simplify models and speed up computation. Principal Component Analysis (PCA) and t-SNE are common techniques.

5. **Anomaly Detection**: The task of identifying unusual patterns or outliers in data. This is important in fraud detection, system health monitoring, and detecting unusual activity in network traffic.

6. **Association Rule Learning**: This involves discovering interesting relations between variables in large databases. It’s commonly used in market basket analysis, cross-selling strategies, and catalog design.

7. **Reinforcement Learning**: Involves training algorithms to make a sequence of decisions by rewarding desired behaviors and/or punishing undesired ones. Applications include robotics, gaming, and navigation.

8. **Natural Language Processing (NLP)**: Tasks in NLP involve understanding, interpreting, and generating human language. Examples include language translation, chatbots, and sentiment analysis.

9. **Recommendation Systems**: These systems predict the preferences or ratings that users would give to a product or service. Widely used in e-commerce, streaming services, and content providers.

10. **Time Series Forecasting**: Involves predicting future values based on previously observed values. Used in stock market analysis, economic forecasting, and weather prediction.
