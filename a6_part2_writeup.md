# Assignment 6 Part 2 - Writeup

---

## Question 1: Feature Importance

Based on your house price model, rank the four features from most important to least important. Explain how you determined this ranking.

**YOUR ANSWER:**
1. Most Important: Square feet
2. Bed room
4. Bath room
5. Least Important: Age

**Explanation:**
This can be easily solved by looking at the coefficent for each of these factors. Square feet had the largest while age had the smllest coefficient



---

## Question 2: Interpreting Coefficients

Choose TWO features from your model and explain what their coefficients mean in plain English. For example: "Each additional bedroom increases the price by $___"

**Feature 1:**
Each additional 1 square foot of living space increases the predicted house price by $121.11,

**Feature 2:**
Each additional bedroom increases the predicted house price by $6,648.97

---

## Question 3: Model Performance

What was your model's R² score? What does this tell you about how well your model predicts house prices? Is there room for improvement?

**YOUR ANSWER:**
The model’s R2 score is 0.9936, which means it explains about 99.36% of the variance in house prices.



---

## Question 4: Adding Features

If you could add TWO more features to improve your house price predictions, what would they be and why?

**Feature 1:**
Lot Size


**Why it would help:**
Larger lots typically increase the value of a property, especially in suburban or rural areas. Including lot size would allow the model to better differentiate between houses with similar square footage but different land sizes.


**Feature 2:**
Location

**Why it would help:**
Why it would help: The location of a house strongly affects its price due to neighborhood quality, proximity to schools, amenities, and local demand. Adding location would capture regional price differences and improve the model’s accuracy

---

## Question 5: Model Trust

Would you trust this model to predict the price of a house with 6 bedrooms, 4 bathrooms, 3000 sq ft, and 5 years old? Why or why not? (Hint: Think about the range of your training data)

**YOUR ANSWER:**
I would be cautious about trusting this prediction because the house is slightly outside the range of the training data (especially the square footage). While the model would provide a reasonable estimate, predictions for houses larger than those in the training set could be less accurate. Models are most reliable when making predictions within the range of data they were trained on.

