# Sentiment Analysis of Sephora Product Reviews

This project analyzes customer reviews of Sephora products to classify their sentiment (positive or negative) and uncover deeper insights into customer preferences and brand perception.

## Project Objectives

- Classify customer reviews as positive or negative using supervised machine learning
- Analyze sentiment trends over time and across brands or categories
- Investigate the impact of review length and product pricing on sentiment
- Explore textual patterns using NLP techniques

## Dataset

The original dataset is sourced from [Kaggle – Sephora Product and Skincare Reviews](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews).

> To reduce repository size, processed CSV files are **not included** in this repo. You can download them here:

- [`reviews_0-250.csv`](https://drive.google.com/file/d/1ZfpER70Kb6y3Fc6SKWkAvLwcFQ7u3X0o/view?usp=sharing)
- [`cleaned_reviews.csv`](https://drive.google.com/file/d/1-4Tml_aMp8GIO_RibIYoqwRSfotHE1M-/view?usp=sharing)

Key features:
- `review_text`: Customer review content  
- `is_recommended`: Target variable (1 = recommended, 0 = not)  
- `rating`, `submission_time`, `total_feedback_count`, `brand`, `category`, etc.

## Models Used

- Logistic Regression  
- Naive Bayes Classifier  
- Random Forest  
- Gradient Boosting

### Text Preprocessing
- Vectorization using **TF-IDF**
- Stopword removal, lowercase normalization, and punctuation cleaning

## Key Insights

- Negative reviews tend to be longer than positive ones
- Higher-priced brands are often associated with more positive reviews
- Clear sentiment trends vary across product categories and over time

## Technologies

- Python (pandas, numpy, sklearn, matplotlib, seaborn)
- Jupyter Notebooks
- NLP with scikit-learn
- TF-IDF for feature engineering

## Author

**Anastasiia Sviridova**  
M.A. in Data Science & Business Analytics, University of Warsaw


