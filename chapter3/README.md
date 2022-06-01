<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>


There are 4 types of Uncertainty Sampling covered in human-in-the-loop:

1. **Least Confidence:** difference between the most confident prediction and 100% confidence:

    $(1-P_\theta(y^*|x))\frac{n}{n-1}$

2. **Margin of Confidence:** difference between the top two most confident predictions:

    $1 - (P_\theta(y_1^*|x) - P_\theta(y_2^*|x))$

3. **Ratio of Confidence:** ratio between the top two most confident predictions:

    $\frac{P_\theta(y_1^*|x)}{P_\theta(y_2^*|x)}$

4. **Entropy:** difference between all predictions, as defined by information theory

    $\frac{-\sum_{y}P_\theta(y|x)log_2 P_\theta(y|x)}{log_2(n)}$

