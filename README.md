# Drawing A Line

The not-so-trivial proces of pattern recognition

## Getting it to work

### Windows
1. Open a command prompt (press your windows key and type 'cmd')
2. Navigate to the root directory of the repository (ie. in DrawALine/) (use 'cd' and 'ls' to navigate in the command prompt)
3. Follow the instructions to: Activate the virtualenv (`.\env\Scripts\activate`)
4. Run the code with ???
5. Add new dependencies from the virtualenv: `pip install (package)`
6. Write out (updated) dependencies: `pip freeze > requirements.txt`

## First time setup
_Windows commands are given, since linux users can probably manage themselves_
1. Install python: https://www.python.org/downloads/
..* Follow the instructions there
2. Visit this page https://packaging.python.org/guides/installing-using-pip-and-virtualenv/
3. Follow the instructions to:
..* Install pip (`py -m pip install --upgrade pip`)
..* Install virtualenv (`py -m pip install --user virtualenv`)
4. Navigate to the root directory of the repository (ie. in DrawALine/) (use 'cd' to navigate in the terminal)
5. Follow the instructions to: Set up a virtualenv (`py -m virtualenv env`)
6. Follow the instructions to: Activate the virtualenv (`.\env\Scripts\activate`)
7. Follow the instructions to: Install the dependencies from the requirements.txt (`pip install -r requirements.txt`)
