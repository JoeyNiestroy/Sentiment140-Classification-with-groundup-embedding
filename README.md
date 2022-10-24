# Sentiment-Classification-with-groundup-embedding
Sentiment Classification model using twitter text corpus and skip gram embedding 
Prelabeled tweets dataset was obatained through kaggle, Pre_Processing.py reads in DF and and normalizes data through character removal, spell check, and lemminization. For ease of post processing changes each step is saved to a new column, all funcions moved to parallel and computing power was used through HPC system thanks to W&M. Sample_Generation.py creates and returns numpy arrays of positive and negative samples (negative samples at 5x rate) using skip-gram sampling, Unigram array was created using techniques from Word2Vec paper, this whole process has also been moved to parallel. Embedding Model was trained across whole data set for one epoch.
This process can be found on Building_Model.py


Embedding_visualization.jpeg is a T-SNE visulization of a small subset of vocabulary with [hi,hello,hey] labeled. 
(See image below)
![Embedding_visualization](https://user-images.githubusercontent.com/106636917/197554027-45fe2ca4-836d-4fef-bf81-6f58868f3f67.jpeg)

Traditional ML models are currently being explored for classification purpose
