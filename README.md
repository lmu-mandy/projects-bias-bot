# Bias Bot Directions

## Step One

- If you have not preprocessed the articles - you must first preprocess them in order to run our code.
  - Create a preprocessed folder, and then a BBC, CNN, and FOX folder within it.
  - If you have to add new samples or articles create a new article preprocessor: 
  ``` preprocess = ArticlePreprocessor(reprocess=True) ```
  - If you don't need to reprocess then create this preprocessor:
  ``` preprocess = ArticlePreprocessor() ```

## Step Two

*Since our model runs on GPU, it is necessary to run it on a colab or a jupyterhub notebook*
- Download the model.ipynb file from our repository
  - First clone our repository
  - Once the repository is downloaded change your directory to our folder
  - Change directory to src
  - Run ```jupyter notebook model.pynb```

## Step Three

- Run all of the cells
- You will be
  - Splitting data into training and testing
  - Building an LSTM model
  - Training the LSTM model in batches with your training data
  - Validating the results of your new model on the testing data

Congratualations! You have completed our tutorial

