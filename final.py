# final.py
# Author: Tyler Stoney
# Class: CSC548 - AI 2
# Professor: Dylan Schwesinger
# Purpose: Machine learn patterns of sentence structures, 
#          print out a response with a structure similar
#          to the structure of an appropriate, real-world
#          response.

from textblob import TextBlob
from fuzzywuzzy import fuzz
import pandas as pd
import re, sys, random, signal
# numpy is never explicitly used, but it's required for textblob to work properly
import numpy
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# 99.99% of data scraped from https://www.eslfast.com/
# The rest came from a zh-en ESL site that I can't find anywhere
# in my browsing history.

class NlpObj(object):
    """
    @brief      an object with a copy of the real string 
                and a string comprised of its words' parts of speech
    """

    main_string = ""
    nlp_string = ""
    def __init__(self, main):
        m = re.split('[?.!]', main)
        tb_main = TextBlob(m[0])
        self.main_string = main
        self.nlp_string = " ".join([x[1] for x in tb_main.tags])

def create_dictionary(lines):
    """
    @brief      Creates a dictionary of statement->response
    
    @param      lines  The lines from the file
    
    @return     the statement/response mapped dictionary
    """

    main_dict = {}

    # For each conversation we read in
    # (conversation defined as a grouping of sentences, 
    # each conversation delimited by an extra newline)
    print("Creating main dictionary...")

    from tqdm import tqdm 
    pbar = tqdm(total=len(lines))
    for convo in lines:

        convo = convo.split('\n')
        if len(convo) < 2:
            continue

        statement = ""
        response  = ""

        # Since not every 'statement' may be on the same line, this function
        #  operates by building upon a string until the next thing it sees
        #  is clearly a new line (beginning with a capital letter).
        have_statement = False
        have_response = False
        building_line = False
        current_line = ""
        line_num = 0
        my_name = ""
        while line_num < len(convo):
            build_count = 0
            current_line = convo[line_num]

            if line_num >= len(convo):
                break

            name_match = re.match(r'^\s*\D\.\s*', convo[line_num])

            line_num+=1
            if name_match:
                my_name = name_match.group(0)

            while line_num < len(convo) and re.match(r'^\s+', convo[line_num]):
                current_line = current_line + " " + convo[line_num].lstrip()
                line_num+=1
                build_count+=1
                building_line = True

            if not have_statement:
                current_line = re.sub(r'^A*B*C*D*E*\.\s+(\.*\(.*\))?\s*', '', current_line)
                statement = current_line
                have_statement = True

            elif not have_response:
                current_line = re.sub(r'^A*B*C*D*E*\.\s+(\.*\(.*\))?\s*', '', current_line)
                response = current_line
                stmt_obj = NlpObj(statement)
                resp_obj = NlpObj(response)

                main_dict[stmt_obj] = resp_obj
                statement = response # once the current line is stored, 
                                     # the current response is used as the new statement
        pbar.update(1)
    pbar.close()
    return main_dict

def create_markov(tags, main_dict, main_dict_keys, main_dict_values):
    """
    @brief      Creates a markov chain of a real word followed by a
                list of potential words categorized by part of speech.
    
    @param      tags       The part of speech tags
    @param      main_dict  the dictionary of statement/response mappings
    
    @return     returns the markov chain and 
                a list of the most frequent words that begin a sentence
    """

    # markkov{Word : POS{[MarkovObj_1, MarkovObj_2, ... MarkovObj_n]}}
    markov = {}

    # most_frequent{POS : freq_list{word : count}}
    most_frequent = {}

    # Create markov chain of words to their most probable part-of-speech
    # candidates
    print("Creating Markov chain...")

    from tqdm import tqdm 
    for key in tqdm(main_dict_keys):
        regex = re.compile('[^a-zA-Z0-9\'\s]')
        sentence_list = regex.sub('', key.main_string).split()
        for i in range(len(sentence_list)-1):
            init_word = sentence_list[i] # the word to begin the markov chain
            pos = TextBlob(sentence_list[i+1]).tags[0][1] # part of speech, to be the key in the new map

            # If it's the first word in the sentence, add its data to the map of most frequent starting words
            if i == 0:
                pos_0 = TextBlob(init_word).tags[0][1]
                if pos_0 in most_frequent:
                    most_frequent[pos_0].append(init_word)
                else:
                    most_frequent[pos_0] = [init_word]

            # Add the word to the list in the markov chain at a position based 
            # on what word precedes it and which part of speech it is.
            if init_word in markov:
                if pos in markov[init_word]:
                    markov[init_word][pos].append(sentence_list[i+1])
                else:
                    markov[init_word][pos] = [sentence_list[i+1]]
            else:
                # Create the markov data for that word if it doesn't exist yet
                markov[init_word] = {}
                markov[init_word][pos] = [sentence_list[i+1]]

    return (markov, most_frequent)


def train_model(tags, main_dict_keys, main_dict_values, df=None):
    """
    @brief      Trains a decision tree based on the structure of the sentences
    
    @param      tags            The part of speech tags
    @param      main_dict_keys  the statements from the statement/response dictionary
    
    @return     returns the trained decision tree and a blank map of all the dataframe headers
    """

    base_count = {}
    for elt in tags:
        base_count[elt] = "" if (len(elt) > 4) else 0

    if not df:
        print("Creating dataframe of sentence structure for training...")

        df = pd.DataFrame(columns=tags)

        from tqdm import tqdm
            
        for s in tqdm(range(len(main_dict_keys))):
            if len(main_dict_values[s].nlp_string) > 0:
                k = main_dict_keys[s].nlp_string.split()
                new_base_count = dict(base_count)
                for elt in k:
                    new_base_count[elt]+=1
                new_base_count['String'] = main_dict_values[s].nlp_string
                df = df.append(new_base_count, ignore_index=True)
        df.to_csv("dataframe.csv", encoding='utf-8', index=False)
    else:
        df = pd.read_csv(df)

    train_set = df.drop(['String'], axis=1)
    answers = df['String']

    print("Training model...")

    model = DecisionTreeClassifier()

    model.fit(train_set, answers)

    return (model, base_count)

def main_loop(model, markov, most_frequent, base_count, tags):
    """
    @brief      prompts user for input, generates a response based on
                machine-learned patterns in sentence structure.
    
    @param      model          The trained decision tree
    @param      markov         The markov chain
    @param      most_frequent  The most frequent starting words
    @param      base_count     A mapping of parts of speech to their frequency in the phrase,
                               used to create the test data for prediction. 
    @param      tags           The part of speech tags
    
    @return     nothing
    """

    while 1:

        try:
            stmt = input("> ")
            if stmt == "quit()":
                break
        except KeyboardInterrupt:
            break

        stmt_obj = NlpObj(stmt)

        test_df = pd.DataFrame(columns=list(tags[:-1]))
        k = stmt_obj.nlp_string.split()
        new_base_count = dict(base_count)
        for elt in k:
            new_base_count[elt]+=1
        new_base_count['String'] = stmt_obj.nlp_string

        test_df = test_df.append(new_base_count, ignore_index=True)
        # print(test_df)
        predictions = []
        predictions = model.predict(test_df.drop(['String'], axis=1))[0].split()
        if len(predictions) == 0:
            print("(EMPTY PREDICTION FOUND)")
            continue
        print(predictions)

        # Start off the response with a random selection of the most frequent sentence-starting word that 
        # matches the part of speech heading the prediction.
        answer = [most_frequent[predictions[0]][random.randint(0, len(most_frequent[predictions[0]])-1)]]

        # current word -> part of speech of the most likely real word to follow it
        # markkov{Word : POS{[MarkovObj_1, MarkovObj_2, ... MarkovObj_n]}}
        # 
        # For the rest of the sentence, I need a reference to the most recently-added word to the response,
        # so I plug that into my markov chain, along with the next part of speech that needs a word, and
        # randomly pick from that underlying list.
        for i in range(0, len(predictions)-1):
            current_word = answer[i]
            # If that part of speech exists as a potential response for the most-recently-added word,
            # pick a random word from the real words available as potential responses.
            try:
                if predictions[i+1] in markov[current_word]:
                    answer.append(markov[current_word][predictions[i+1]][random.randint(0, len(markov[current_word][predictions[i+1]])-1)])
                else:
                    # otherwise, fuzzy search for similar parts of speech that are available as responses
                    # and pick a real word form those available for that part of speech. 
                    keys = list(markov[current_word].keys())
                    longest = 0
                    longest_ratio = fuzz.ratio(stmt, keys[0])
                    for s in range(len(keys)):
                        test_ratio = fuzz.ratio(stmt, predictions[i+1])
                        if test_ratio > longest_ratio:
                            longest_ratio = test_ratio
                            longest = s
                    answer.append(markov[current_word][keys[longest]][random.randint(0, len(markov[current_word][keys[longest]])-1)])
            except KeyError:
                print("Couldn't find a match following the word '" + current_word + "'")
                pass
        print(' '.join(answer))

        # This was the old, original version that I displayed in my email; 
        # rather than the convoluted version you see today, I just
        # fuzzy-searched for a key in my statement-response map that was
        # the most similar to what the user entered, then printed out
        # the response that matched. Naturally this led to real sentences
        # that were gramatically correct, but my current method is more fun.
        # 
        # longest = 0
        # main_dict_keys = list(main_dict.keys())
        # main_dict_values = list(main_dict.values())
        # longest_ratio = fuzz.ratio(stmt, main_dict_values[0].nlp_string)
        # for s in range(len(main_dict_values)):
        #     test_ratio = fuzz.ratio(stmt, main_dict_values[s].nlp_string)
        #     if test_ratio > longest_ratio:
        #         longest_ratio = test_ratio
        #         longest = s
        # print(main_dict_values[longest].main_string)
        # print(main_dict_values[longest].nlp_string)
        pass 

def main():

    df = None
    if len(sys.argv) == 2:
        df = sys.argv[1]
    try:
        pd.read_csv(df)
    except:
        print("'" + df + "' is not a good data file.")
        sys.exit(1)

    tags = ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBZ','VBP','VBD','VBN','VBG','WDT','WP','WP$','WRB','NP','PP','VP','ADVP','ADJP','SBAR','PRT','INTJ','PNP','String']

    lines = []
    with open('convos_1.txt') as f:
        lines = f.read().split('\n\n')

    
    main_dict = create_dictionary(lines)

    main_dict_keys = list(main_dict.keys())
    main_dict_values = list(main_dict.values())

    mark = create_markov(tags, main_dict, main_dict_keys, main_dict_values)
    markov = mark[0]
    most_frequent = mark[1]

    mod = train_model(tags, main_dict_keys, main_dict_values, df)
    model = mod[0]
    base_count = mod[1]

    main_loop(model, markov, most_frequent, base_count, tags)

if __name__ == '__main__':
    main()