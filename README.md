**Smart Grid Stability Model**
This repository contains a machine learning model and a graphical user interface (GUI) designed to predict and explain the stability of a smart grid. The model uses various features related to power consumption and reaction times to determine whether a grid is stable or unstable. The project also includes a Tkinter-based GUI for easy file uploads and real-time predictions.

**Project Overview**
The Smart Grid Stability Model predicts whether a smart grid is stable or unstable based on data such as power consumption, response times, and elasticity coefficients. The model is built using a deep neural network and is capable of providing insights into the factors affecting grid stability. A user-friendly interface allows users to upload CSV files containing grid data, view predictions, and get explanations for the stability status.

**Features**
Deep Neural Network Model: Trained to predict grid stability based on input features such as power consumption, reaction times, and price elasticity.
Data Preprocessing: Scales input data using a pre-fitted StandardScaler to ensure accurate predictions.
Explanatory Insights: Provides detailed explanations of the model’s decision-making process, outlining factors like high power consumption, slow response times, and low price elasticity that contribute to instability.
GUI for Predictions: A Tkinter-based GUI that allows users to upload CSV files, view predictions, and read explanations in an easy-to-use text box.

**Model Explanation**
The model works by processing four primary types of features:
Reaction times (_tau1, tau2, tau3, tau4_): Delays in response times from suppliers and consumers, affecting stability.
Power consumption (_p1, p2, p3, p4_): The power consumed by various consumers and its impact on overall balance.
Price elasticity coefficients (_g1, g2, g3, g4_): The responsiveness of consumers and the supplier to price changes, influencing demand and supply stability.

If the model predicts instability, it will provide an explanation highlighting which features are contributing to the issue, such as high consumption, delayed responses, or low elasticity.
