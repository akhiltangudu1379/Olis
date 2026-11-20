---
title: Olist Cancellation Prediction API
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Olist Cancellation Prediction API

FastAPI deployment of an XGBoost DART cancellation classifier. Visit `/docs` for interactive Swagger API documentation.

The model is an educational classifier trained on historical Olist order data. Its outputs are not guaranteed probabilities of future cancellation, and should not be used as operational decisions without further validation.
