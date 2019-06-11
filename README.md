(THE FOLLOWING README WAS HANDED IN AS PART OF THE PROJECT)

RUN WITH PYTHON 3

PACKAGES TO INSTALL
===================
textblob  
fuzzywuzzy
pandas
sklearn
numpy
tqdm

you can easily do this by running my install packages script :)  
probably needs root access, though. Maybe my calls to 
pip need root access as well, not just the script.

ALSO NEEDED (for textblob creation)
=================
python -m textblob.download_corpora

RUNNING THE PROGRAM
===================
My package scanner/installer takes in a list of file names as arguments. Easy enough.

My real project is easy enough as well: if no argument is provided, the program will
create a dataframe from the massive amount of data it has in memory.  This is time
consuming (takes ~5 minutes on my laptop).  It will save it as a csv for future use,
however.  I provided the csv, but if you have doubts or just wanna run it fresh
yourself, go run and get something to eat in the meantime.  Fair warning.

python3 final.py <OPTIONAL NAME OF DATAFRAME.csv>


HOW IT "WORKS"
==============
- A dictionary of sentences/nlp strings is made, mapping a statement to a response
- A Markov chain is made in a multimap-esque structure, where the initial key is a real word
  and it is mapped onto a list of POS tags, each of which is a key to another map to a list of real
  words that follow the original key.
- A dataframe is created (if not user-supplied) where the stats of the parts of speech per statement
  are tallied, and associated with the nlp string associated with the statement's response (from the
  main dictionary)
- A decision tree is trained on this dataframe.
- The main loop starts up, prompting the user for input
- The user's input is tokenized and shoved into the model for prediction
- When the model finds a prediction, the Markov chain springs into action and populates the
  response with appropriate words that fit the current part-of-speech in the phrase that most closely
  follows the previously selected word.
- Repeat until user enters 'quit()' or hits CTRL+C

BUGS
====
- Sometimes it can't find a word in the markov chain, so it crashes. 
(Should be handled as of 05-05-18)
This is probably due to the off-by-one model of the keys vs the 
values in the main dictionary: on the slim (but apparently existent) 
chance that a word exists in one but not the other, it'll crash.

- Wouldn't call it a bug, but it usually doesn't generate sentences that
are gramatically correct.
