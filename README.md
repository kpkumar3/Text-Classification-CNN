#Text-Classification-CNN
#Code is written specific to our project requirements.
News articles are extracted every day from a source (with proper authorization) and the raw data is stored in Hadoop. 
Basic functionality: Code reads the news article data from Hadoop, cleans it up and classifies the text as relevant/irrelevant
In the first run, the code builds, trains a model on manually labelled data. The trained model is used to classify the new texts extracted from our source.

Data:
Mastercopy.csv contains initial training data. 
Other requirements: 
Code uses Global Vectors for Word Representation(GloVe) to create the embedding layer.
Download http://nlp.stanford.edu/data/glove.6B.zip and extract the contents. Use glove.6B.100d.txt for 100 dimension representation of 40000 vectors.

Automation: 
Code is automated to run every day at 8AM. 
If the new file is not found, code keeps running once in every two hours until 10PM to check if the file has been created. 
If the file is processed successfully, the code sleeps until 8AM next morning.
If the file has no data, the code sleeps until 8AM next morning.

Every sunday, the code uses new data (if any) and retrains the model.
