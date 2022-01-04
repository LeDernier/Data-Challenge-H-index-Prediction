# Data-Challenge-H-index-Prediction

### Introduction

This project was developed during the INF554 - Machine and Deep Learning course at Ecole Polytechnique during the period 2021-2022. It was a private Kaggle challenge where the objective was to find the best model, based on the MSE (Mean Squared Error), to predict the h-index of the authors in the computer science domain. More details about the project are available in the document **INF554_data_challenge.pdf**.

### Install

This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [langdetect](https://pypi.org/project/langdetect/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [pytorch](https://pytorch.org/)
- [NLTK](https://www.nltk.org/)
- [Gensim](https://radimrehurek.com/gensim/index.html)
- [WordCloud] (https://pypi.org/project/wordcloud/)
- [lightgbm](https://lightgbm.readthedocs.io/en/latest/)
- [xgboost](https://xgboost.readthedocs.io/en/stable/)


You will also need to have software installed to run and execute a [Jupyter Notebook](http://jupyter.org/install.html).

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](https://www.anaconda.com/download/) distribution of Python, which already has the above packages and more included. 

### Getting started
Download all data provided for the challenge and place them inside the directory `Data`.
This includes the `abstracts`,`coauthorship.edgelist`,`train.csv`,`test.csv`,and `author papers.txt`

### Code

The `Data preprocessing.ipynb` file in inside is use to process the abstracts data. Check this file on how we process the abstracts to generate features to train the model.
The `Graph features`  and `node_embedding` notebook files were used to process the graph related data.

### Run

In oder to run the `models` notebook,please ensure that you have all the data and has run the notebook file inside `Data` directory and the notebook files inside `Graph features` folder.
In bash:

```bash
ipython notebook notebook_file_name.ipynb
```  
or
```bash
jupyter notebook notebook_file_name.ipynb
```
or open with Jupyter Lab
```bash
jupyter lab
```

This will open the Jupyter Notebook software and project file in your browser.

### Data

The features we generated after processing for training the model as follows
`
['n_embedding_0', 'n_embedding_1', 'n_embedding_2', 'n_embedding_3', 'n_embedding_4', 'n_embedding_5', 'n_embedding_6', 'n_embedding_7', 'n_embedding_8', 'n_embedding_9', 'n_embedding_10', 'n_embedding_11', 'n_embedding_12', 'n_embedding_13', 'n_embedding_14', 'n_embedding_15', 'n_embedding_16', 'n_embedding_17', 'n_embedding_18', 'n_embedding_19', 'at_embedding_0', 'at_embedding_1', 'at_embedding_2', 'at_embedding_3', 'at_embedding_4', 'at_embedding_5', 'at_embedding_6', 'at_embedding_7', 'at_embedding_8', 'at_embedding_9', 'at_embedding_10', 'at_embedding_11', 'at_embedding_12', 'at_embedding_13', 'at_embedding_14', 'at_embedding_15', 'at_embedding_16', 'at_embedding_17', 'at_embedding_18', 'at_embedding_19', 'at_embedding_20', 'at_embedding_21', 'at_embedding_22', 'at_embedding_23', 'at_embedding_24', 'at_embedding_25', 'at_embedding_26', 'at_embedding_27', 'at_embedding_28', 'at_embedding_29', 'at_embedding_30', 'at_embedding_31', 'at_embedding_32', 'at_embedding_33', 'at_embedding_34', 'at_embedding_35', 'at_embedding_36', 'at_embedding_37', 'at_embedding_38', 'at_embedding_39', 'at_embedding_40', 'at_embedding_41', 'at_embedding_42', 'at_embedding_43', 'at_embedding_44', 'at_embedding_45', 'at_embedding_46', 'at_embedding_47', 'at_embedding_48', 'at_embedding_49', 'at_embedding_50', 'at_embedding_51', 'at_embedding_52', 'at_embedding_53', 'at_embedding_54', 'at_embedding_55', 'at_embedding_56', 'at_embedding_57', 'at_embedding_58', 'at_embedding_59', 'at_embedding_60', 'at_embedding_61', 'at_embedding_62', 'at_embedding_63', 'core_number', 'clustering_coef', 'betweeness_coef', 'centrality', 'page_rank', 'clustering_coef_coauthorship', 'centrality_coauthorship', 'page_rank_coauthorship', 'degree', 'onion_number', 'weighted_degree' `
 
**Target Variable**
4. `h-index`: the h-index of authors

**Model Evaluation Metric**
Mean squared error
**BEST MODEL**
lightgbm model gave the best performance with MSE of 49.64101


