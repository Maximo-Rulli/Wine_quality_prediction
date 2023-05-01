# Wine Quality Classification with Softmax Regression
This is a project that uses Softmax Regression to classify the quality of wines based upon their features. The model was trained on a dataset containing several variables such as fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol.

## Project Structure
The project has the following files:

- winequality.csv: Dataset used for training the model.
- softmax_regression.py: Python script containing the Softmax Regression algorithm and the necessary code for training and evaluating the model.
- parameters: Pre-trained parameters used for classification of the wine qualities.
- README.md: Markdown file containing project information and instructions.
- images/: Folder containing the images used in the project.
- Emoji0.png, Emoji1.png, Emoji2.png, Emoji3.png, Emoji4.png, Emoji5.png, Emoji6.png, Emoji7.png, Emoji8.png, Emoji9.png: Images used for visualization of the results.
- requirements.txt: Text file containing the necessary packages to be installed for running the project.

## Installation
. Clone the repository: ``` git clone https://github.com/Maximo-Rulli/Wine-quality-prediction.git ```

. Navigate to the project directory: ``` cd Wine-quality-prediction ```
. Install the required packages: ``` pip install -r requirements.txt ```
. Run the softmax_regression.py script: ``` python softmax_regression.py ```


## Usage
. Run the softmax_regression.py script.
. Adjust the sliders to set the wine features.
. The predicted wine quality will be displayed as a point on the interactive graph.

## Interactive Graph
An interactive graph was developed to display the predicted wine quality as a point on a scatterplot. The graph displays ten emojis, each corresponding to a quality score from 1 to 10. The x-axis and y-axis represent the predicted quality score and the actual quality score, respectively. The graph is interactive and updates in real-time as the user adjusts the sliders to set the wine features.

Interactive Graph

Conclusion
This project demonstrates the use of Softmax Regression for wine quality classification based on different features. The model achieved good accuracy and the interactive graph allows for easy visualization of the results.
