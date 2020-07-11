For this project we used data from a Datadriven compettition. To get access to the data use this [link](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/). You will need to create an account to gain access to the data download and helpful dicussion boards.


The [Water Ministry](http://maji.go.tz/) can be found at this website to give more background on the situtation with the wells of Tanzania as well as other local news.

The Organization that aided in collecting thid ata is [Tariffa](http://taarifa.org/). They are a multi nation organization helping to aid the world and be a creat effective channels of communication for developing worlds.

After you gain access to the data. This was as [helpful site on total static head and general pump information](https://www.pumpfundamentals.com/what%20is%20head.htm)

Lasty this folder also contains a pdf of the meanings of the columns found in the discussions section of the Tanzania water well project dicussion section.





















# Modeling
To begin our modeling we got an overall picure of our data and saw many duplicates and missing values so we did what we could to impute where it was deemed neccesary, and worked on narrowing down on our meaningful feature. To do this we created a basic decision tree to help see how our tree was being split. Now that we had our deemed features of value we then looked at our class types of our wells. This then led to us seeing an inbalance in our data and to fix this we used SMOTE or weighted out classes. This was followed with going through mutliple algorithms to select the best one.

#  Model Evaluation
In order to determine the effectiveness of our model we focused on the recall score of the non functionaing pumps. We would run our trained model on our test data to see how well it preformed when given novel data.\

# Discussion of deployment
To deploy this model we would implement it in the Water Ministry of Tanzania to help better predict the wells given a set of features specefied within our model. This will hopefully aid in the development of Tanzania and bring value to their country.






#### Repo Navigation Links
 - [final summary notebook]()-Summary of our project's code and walking through our process.
 - [exploratory notebooks folder](https://github.com/KarimOliver/Tanzania/tree/master/exploratory)-The code we used to acheive the results displayed in this repository. 
 - [src folder](https://github.com/KarimOliver/Tanzania/tree/master/src/data_cleaning)-Python modules used to clean and analyze data.
 - [references](https://github.com/KarimOliver/Tanzania/tree/master/references)-Useful information we found and utilized to get a clear understanding of the data.
 
 
 
 
 
 
 
 
 
 
 
# General Setup Instructions 

Ensure that you have installed [Anaconda](https://docs.anaconda.com/anaconda/install/) 

### `Tanzania` conda Environment

This project relies on you using the [`Tanzania.yml`], or [`Tanzania_environment_for_windows.yml`](Tanzania_environment_for_windows.yml) file to recreate the `Tanzania` conda environment. To do so, please run the following commands *in your terminal*:
```bash
# create the housing conda environment
Mac operating system:
conda env create -f Tanzania.yml
Windows operating system:
conda env create -f Tanzania_environment_for_windows.yml
# activate the housing conda environment
conda activate Tanzania
# if needed, make housing available to you as a kernel in jupyter
python -m ipykernel install --user --name Tanzania --display-name "Python 3 (Tanzania)"
```

conda activate Tanzania
conda env export > environment.yml