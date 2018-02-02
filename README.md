# Drawing A Line

The not-so-trivial process of pattern recognition

[![Build Status](https://travis-ci.com/i3anaan/DrawALine.svg?token=idmHsk2qU3ZuQEw2J112&branch=master)](https://travis-ci.com/i3anaan/DrawALine)

## First time Installation
_Windows commands are given, since linux users can probably manage themselves_
1. Install python: https://www.python.org/downloads/
    * Follow the instructions there
3. Follow [these instructions](https://packaging.python.org/guides/installing-using-pip-and-virtualenv/) to:  to:
    1. Install pip (`py -m pip install --upgrade pip`)
    2. Install virtualenv (`py -m pip install --user virtualenv`)
4. Navigate to the root directory of the repository (ie. in DrawALine/)
    * _(use `cd` to navigate in the terminal)_
5. Do the following:
    1. Set up a virtualenv (`py -m virtualenv env`)
    2. Activate the virtualenv (`.\env\Scripts\activate`)
    3. Install the dependencies from the requirements.txt (`pip install -r requirements.txt`)

## Usage
_Please follow the first time installation steps before usage_
1. Activate the virtualenv (`.\env\Scripts\activate`)
2. Run the program using (`python .\src\main.py [preprocessing] [classifier] [parameters]`)
    * Preprocessing options are used to control the dataset that is fed into the classifier
    * The classifier option determines which classifier to used
    * Each classifier can have their own settings, set using options after the classifier
        * **Note that default settings will be used for parameters that are not set, this can include iterating over a range.**
4. Use the `-h` option for more information or see below for examples

### Full help information
`python3 ./src/main.py -h`
```
usage: DrawALine [-h] [--test-run] [--small] [--distort {shift,grow,all}]
                 [--similarity {dsim_edit,sim_norm1,sim_norm2,sim_cos}]
                 [--pca PCA] [--test-set {default,eval,live}]
                 [--digits-per-class DIGITS_PER_CLASS]
                 {mlp,knn,svc,qda,lda,log} ...

Pattern Recognition tool for recognizing decimals from the NIST dataset.

positional arguments:
  {mlp,knn,svc,qda,lda,log}
                        classifiers

optional arguments:
  -h, --help            show this help message and exit
  --test-run            Run in implementation test mode - use a tiny dataset
  --small               Use a small training set
  --distort {shift,grow,all}
                        Distort the data
  --similarity {dsim_edit,sim_norm1,sim_norm2,sim_cos}
                        Transform the data to similarity representation
  --pca PCA             Use PCA feature extraction
  --test-set {default,eval,live}
                        Test on a seperate dataset (uses the entire default
                        data-set for training)
  --digits-per-class DIGITS_PER_CLASS
                        The number of digits per class to use for testing in
                        evaluation mode

```
`python3 ./src/main.py svc -h`
```
usage: DrawALine svc [-h] [--gamma GAMMA] [--C C]
                     [--kernel {linear,poly,rbf,sigmoid,precomputed}]

optional arguments:
  -h, --help            show this help message and exit
  --gamma GAMMA         The gamma setting
  --C C                 The C setting
  --kernel {linear,poly,rbf,sigmoid,precomputed}
                        The kernel setting

```

## Output
This program outputs in 2 ways, on the terminal and by making use of `.csv` log files.
  * **The terminal** will give some progress messages as well as confirming most of the preprocessing and parameters set actually being used. Additionally it will output the accuracy (on the test and training data) and time (test time and training time).
  * **CSV Log files** are created and used for every run of every classifier. These will be created in the `./results/` folder and are categorized by classifier. Each row will contain information of 1 run, including all classifier options set and general results such as accuracy and timing.

### Example terminal output
`python3 src/main.py --small --distort=shift --pca=45 svc --gamma=3`
```
Cherry picking dataset...
Applying shift distortion...
Applying PCA feature extraction...
Training set size: (500, 45)
Test set size:     (9900, 45)
Overriding settings...
{'gamma': 3.0}
#>SVC<# Train/Test: 100.00%/010.00%  Train/Test: 0.0578/0.5076
#>SVC<# Train/Test: 100.00%/010.00%  Train/Test: 0.0417/0.5294
#>SVC<# Train/Test: 100.00%/010.00%  Train/Test: 0.0363/0.3941
#>SVC<# Train/Test: 100.00%/010.00%  Train/Test: 0.0359/0.3936
#>SVC<# Train/Test: 100.00%/010.00%  Train/Test: 0.0378/0.4371
#>SVC<# Train/Test: 100.00%/010.00%  Train/Test: 0.0365/0.3893
#>SVC<# Train/Test: 100.00%/010.00%  Train/Test: 0.0367/0.3885
#>SVC<# Train/Test: 100.00%/010.00%  Train/Test: 0.0362/0.3927
#>SVC<# Train/Test: 100.00%/010.00%  Train/Test: 0.0411/0.5204
#>SVC<# Train/Test: 100.00%/010.00%  Train/Test: 0.0423/0.5060
#>SVC<# Train/Test: 100.00%/010.00%  Train/Test: 0.0420/0.5080
```
