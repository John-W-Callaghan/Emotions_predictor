# Emotion Predictor

This Python program predicts emotions (sadness, joy, love, anger, fear, surprise) based on user input sentences. It uses a logistic regression model trained on a dataset containing labeled text data.

## Getting started
 - Prerequisites
 - Python 3.x

Required Python packages: pandas, scikit-learn, tkinter
Install the required packages using pip:

```bash
pip install pandas scikit-learn
```


## Usage
1. Clone the repository
```python
git clone https://github.com/yourusername/emotion-prediction.git
```
2. Navigate to the project directory
```python
cd emotion-prediction

```
3. Run the Python script:
```python
python emotion_prediction.py

```

Enter a sentence in the input field and click the "Predict Emotion" button to see the predicted emotion.

Repeat step 4 to predict emotions for more sentences.

## Description

The program uses a logistic regression model trained on a dataset containing labeled text data. It preprocesses the input text data using a CountVectorizer and predicts the emotion associated with the input sentences.

## Files

- `emotion_prediction.py:` Main Python script for the emotion prediction program.
- `text.csv:` Dataset containing labeled text data for training the logistic regression model.
- `README.md:` This README file providing information about the project.
- `tokenise.py:` Python script for tokenisastion and preprocesses
## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
