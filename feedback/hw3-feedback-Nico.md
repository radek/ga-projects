### ***Project 3 Feedback***

***Nico Van de Bovenkamp***

***

**Overall:**  
Great work on this assignment! You really through the whole bucket at it, huh? Nice job! You tried pretty much every single model that we learned in class up to that point. Awesome! You efficiently worked through the data, and appropriately scaled your X and built some awesome classifiers. It is really amazing how well that KNN worked. It looks like the points cluster quite geometrically. Your last model, the `SGDClassifier` worked pretty well too given that it is also distanced based. Awesome! I encourage you to really think about ***why*** those models seemed to perform so well, and maybe some of the others slightly less so.

I have a few thoughts/comments on your modeling process below. Some of them are on fundamentals, others are on some more "best practices".

**Some notes**  

* **Train/test Split earlier** As we discussed in our review session, the first thing that you want to do is perform your train/test split (once you have your data). You don't want any information about your test set to "leak" into what you are learning about your training set. All preprocessing, transformations, handling of missing values, etc. should be done on your train set as the goal is to have it generalize onto your test set. Again, the concept is: "We don't have the test yet, that comes tomorrow." So, for example, when you scaled your X, you should only be scaling according to your X_train:

```python
x_train_scaled = ss.fit_transform(X_train)
x_test_scaled = ss.transform(X_test)
```

* **Types of models**  As we discussed in the review session, certain models are constructed for certain tasks. Part of the reason your Lasso and Ridge models did not work well is because they are not appropriate for the given task. Those models are regularized flavors of our Linear Regression model, which is used for regression tasks. The other models (KNN, LogisticRegression, SGDClassifier) performed well because they are mathematically/algorithmically suited for the classification task of skin 1 or skin 2.
* **Hyperparameter optimization**  As we discussed in our review session, once you have decided on a solid set of features, you want to do some level of hyper parameter tuning. You currently aren't adding any penalty to your model, which could help with some over-fitting (though, honestly, there isn't much overfitting going on. You can tell because your out-of-sample error on your cross-validations are not much lower than your train errors.).
    - Logistic regression:
        - C : Amount of penalty you apply to your weights! Remember that in this implementation, it is actually 1/C. Thus, a smaller C means more penalty as you are adding MORE of the weights to the penalty.
        - penalty: 'l1' vs. 'l2'
        - fit_intercept : (this is a parameter, but you likely always use this term)
        - solver : {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’} this is your term to change the optimization procedure! These can be interesting to experiment with. On a smaller dataset, this may not be so important, but when datasets become huge... This is very important.
* **Classification metrics**  Just as we have different models for different tasks, we have different evaluation metrics for different tasks. R2 score, MSE, RMSE, and MAE (mean absolute error) are regression metrics. They should not be used in this classification task. Your other metrics are great!
    - Check out this list: http://scikit-learn.org/stable/modules/classes.html
* **A note on predict methods**  In Sklearn, there are a few ways that we can "predict" in the api. The methods given are always defined within the documentation, as they are not always the same. For LogisticRegression, we have a `predict()`, `predict_proba()`, `predict_log_proba()`, and `decision_function()`. If you look through the documentation, you will notice that these functions output/mean different things. You have used the `decision_function()`, which outputs a "confidence score" by taking the distance of the given point from the hyper-plane (the plane that is separating our classes). A positive score indicates a high confidence via distance, and a low score would be greatly negative as it is on the "wrong side" of the prediction boundary. You can use this, but be aware of what it truly means! I would recommend using probabilities with Logistic Regression, as it's one of the benefits of having this model in the first place! `predict()` will predict the **class** and 'predict_proba()' will predict a probability.
