# History:
- Tried a baseline byt5 model 
- Tried adapting a different arabian-pretrained model
    - MODEL_NAME = "Helsinki-NLP/opus-mt-tc-big-ar-en"
    - Found that the results were poor. Went with byt5 instead
        - Because another contestant had provided code using it and it worked well.
        - Additionally the byt5 is trained on like 100 languages so maybe arabic is in there.
- Tried making dictionary data into a direct one to one training set. 
    - It resulted in much worse results than the default untrained byt5 model.
    - My guess is that this is because the transformer cannot make use of its relationship-based architecture when training on a dictionary.
    - I hope to later use the contents to augment the data with some word substitutions using the english side of it.

# Current Plans:
- (Training Debugging) Attempting masked language modelling technique
    - Mentioned in the book I'm reading called Natural Language Processing with Transformers as a way to make use of unlabelled data.
    - ran into errors because byt5 works differently from BERT. And thus needs some special data ??? things to arrange the data.
- (Data Prep) Translate that Akkad->German data to be english on the german side instead. 
    - I will use the byt5 for this since it likely does well already.
- (Ensembling) Once I feel I have explored what possibilities sound fun, I want to try ensembling.
    - K-fold cross validation, possibly with Model Soup afterwards sounds good.
- (Data Augmentation) What if I made use of the specificity of the data?
    - Since a lot of the data is like a business email, what if I swapped out names, swapped out beginning address lines, etc.? Maybe that is possible.
    - There are a lot of numbers, I can swap numbers.
- TODO ADD EXCEL EXPORT FOR 10 random (fixed seed) data points from the validation set of the training set. So that way I can see qualitatively how the model is doing, along with seeing a correct answer.


# Current Problems I want to solve:
- The data size is small. I want to augment it.
    - The authors have provided datasets: 
        - a Akkad->German dataset
        - Unlabelled data
    - The book I am reading called Natural Language Processing with Transformers mentions a few techniques I can use to augment data, or to make use of unlabelled data, and I want to try those out.
- Apparently the data is a bit unclean? I am ignoring this.
    - One example is that the numbers in the data need to be messed with or made the same, etc.
- 