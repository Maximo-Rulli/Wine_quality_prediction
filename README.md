# Wine Quality Classification with Softmax Regression
This is a project that uses Softmax Regression to classify the quality of wines based upon their features. The model was trained on a dataset containing several variables such as fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol.

## Project Structure
The project has the following files:

- winequality.csv: Dataset used for training the model.
- graph.py: Python script containing the Softmax Regression algorithm and the necessary code for displaying the predictions.
- Wine_quality.ipynb: Jupyter notebook with the necessary code to train the Softmax Regression algorithm and save the parameters.
- parameters: Pre-trained parameters used for classification of the wine qualities.
- README.md: Markdown file containing project information and instructions.
- Emoji0.png, Emoji1.png, Emoji2.png, Emoji3.png, Emoji4.png, Emoji5.png, Emoji6.png, Emoji7.png, Emoji8.png, Emoji9.png: Images used for visualization of the results.
- requirements.txt: Text file containing the necessary packages to be installed for running the project.

## The dataset
The dataset was downloaded from Kaggle and it has 1599 examples of wines with qualities ranging from 3 to 8. Each wine had its own features as mentioned earlier. You can download the original dataset from the following link:
https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009

## Installation
1. Clone the repository: ``` git clone https://github.com/Maximo-Rulli/Wine-quality-prediction.git ```
2. Navigate to the project directory: ``` cd Wine-quality-prediction ```
3. Install the required packages: ``` pip install -r requirements.txt ```
4. Run the graph.py script: ``` python graph.py ```


## Usage
1. Run the graph.py script.
2. Adjust the sliders to set the wine features.
3. The predicted wine quality will be displayed as a point on the interactive graph.

## Retraining
The model can be retrained to obtain different results, to achieve this, run the Wine_prediction.ipynb: ``` jupyter notebook Wine_prediction.ipynb ```

## Interactive Graph
An interactive graph was developed to display the predicted wine quality as a point on a scatterplot. The graph displays ten emojis, each corresponding to a quality score from 1 to 10. The x-axis and y-axis represent the predicted quality score based on the sliders inputs. The graph is interactive and updates in real-time as the user adjusts the sliders to set the wine features.

![Screenshot of the interactive graph generated when running graph.py](/screenshot.png)

## Conclusion
This project demonstrates the use of Softmax Regression for wine quality classification based on different features. The model achieved a good loss (1.16528 on sparse categorical crossentropy) and the interactive graph allows for easy visualization of the results.
