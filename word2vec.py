import os, sys, re, csv
import pickle
from collections import Counter, defaultdict
import numpy as np
import scipy
import math
import random
import nltk
from nltk.corpus import stopwords

nltk.download
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
from numba import jit

# ... (1) First load in the data source and tokenize into one-hot vectors.
# ... Since one-hot vectors are 0 everywhere except for one index, we only need to know that index.


# ... (2) Prepare a negative sampling distribution table to draw negative samples from.
# ... Consistent with the original word2vec paper, this distribution should be exponentiated.


# ... (3) Run a training function for a number of epochs to learn the weights of the hidden layer.
# ... This training will occur through backpropagation from the context words down to the source word.


# ... (4) Re-train the algorithm using different context windows. See what effect this has on your results.


# ... (5) Test your model. Compare cosine similarities between learned word vectors.










# .................................................................................
# ... global variables
# .................................................................................


random.seed(10)
np.random.seed(10)
randcounter = 10
np_randcounter = 10

vocab_size = 0
hidden_size = 100
uniqueWords = [""]  # ... list of all unique tokens
wordcodes = {}  # ... dictionary mapping of words to indices in uniqueWords
wordcounts = Counter()  # ... how many times each token occurs
samplingTable = []  # ... table to draw negative samples from

# .................................................................................
# ... load in the data and convert tokens to one-hot indices
# .................................................................................

'''
def better_tokenize(line):
    EN_WHITELIST = 'abcdefghijklmnopqrstuvwxyz '
    stopwords_list = []
    tokenized_text = []
    for token in ''.join([ch for ch in line.lower() if ch in EN_WHITELIST]).split(" "):
        if token in stopwords_list:
            continue
        if token:  # Removes empty elements
            tokenized_text.append(token)
    return tokenized_text
'''


def loadData(filename):
    global uniqueWords, wordcodes, wordcounts
    override = True
    if override:
        # ... for debugging purposes, reloading input file and tokenizing is quite slow
        # ...  >> simply reload the completed objects. Instantaneous.
        fullrec = pickle.load(open("w2v_fullrec.p", "rb"))
        wordcodes = pickle.load(open("w2v_wordcodes.p", "rb"))
        uniqueWords = pickle.load(open("w2v_uniqueWords.p", "rb"))
        wordcounts = pickle.load(open("w2v_wordcounts.p", "rb"))
        return fullrec

    # ... load in the unlabeled data file. You can load in a subset for debugging purposes.
    handle = open(filename, "r", encoding="utf8")
    fullconts = handle.read().split("\n")
    fullconts = [entry.split("\t")[1].replace("<br />", "") for entry in fullconts[1:(len(fullconts) - 1)]]

    # ... apply simple tokenization (whitespace and lowercase)
    fullconts = [" ".join(fullconts).lower()]

    print("Generating token stream...")
    # ... (TASK) populate fullrec as one-dimension array of all tokens in the order they appear.
    # ... ignore stopwords in this process
    # ... for simplicity, you may use nltk.word_tokenize() to split fullconts.
    # ... keep track of the frequency counts of tokens in origcounts.

    #word_tokens = better_tokenize(fullconts[0])
    word_tokens = nltk.word_tokenize((fullconts[0]))
    stop_words = set(stopwords.words('english'))
    min_count = 50

    fullrec = [w for w in word_tokens if not w in stop_words]
    origcounts = Counter(fullrec)

    print("Performing minimum thresholding..")
    # ... (TASK) populate array fullrec_filtered to include terms as-is that appeared at least min_count times
    # ... replace other terms with <UNK> token.
    # ... update frequency count of each token in dict wordcounts where: wordcounts[token] = freq(token)
    fullrec_filtered = [w if origcounts[w] >= min_count else "<UNK>" for w in fullrec]

    # ... after filling in fullrec_filtered, replace the original fullrec with this one.
    fullrec = fullrec_filtered
    wordcounts = Counter(fullrec)

    print("Producing one-hot indicies")
    # ... (TASK) sort the unique tokens into array uniqueWords
    # ... produce their one-hot indices in dict wordcodes where wordcodes[token] = onehot_index(token)
    # ... replace all word tokens in fullrec with their corresponding one-hot indices.
    uniqueWords = set(fullrec)  # ... fill in
    for i, word in enumerate(uniqueWords):
        wordcodes[word] = i

    fullrec = list(map(lambda x: wordcodes[x], fullrec))

    # ... close input file handle
    handle.close()

    # ... store these objects for later.
    # ... for debugging, don't keep re-tokenizing same data in same way.
    # ... just reload the already-processed input data with pickles.
    # ... NOTE: you have to reload data from scratch if you change the min_count, tokenization or number of input rows


    pickle.dump(fullrec, open("w2v_fullrec.p", "wb+"))
    pickle.dump(wordcodes, open("w2v_wordcodes.p", "wb+"))
    pickle.dump(uniqueWords, open("w2v_uniqueWords.p", "wb+"))
    pickle.dump(dict(wordcounts), open("w2v_wordcounts.p", "wb+"))

    # ... output fullrec should be sequence of tokens, each represented as their one-hot index from wordcodes.
    return fullrec


# .................................................................................
# ... compute sigmoid value
# .................................................................................
@jit(nopython=True)
def sigmoid(x):
    return float(1) / (1 + np.exp(-x))


# .................................................................................
# ... generate a table of cumulative distribution of words
# .................................................................................


def negativeSampleTable(train_data, uniqueWords, wordcounts, exp_power=0.75):
    # global wordcounts
    # ... stores the normalizing denominator (count of all tokens, each count raised to exp_power)
    max_exp_count = 0

    print("Generating exponentiated count vectors")
    # ... (TASK) for each uniqueWord, compute the frequency of that word to the power of exp_power
    # ... store results in exp_count_array.
    exp_count_array = [math.pow(wordcounts[t], exp_power) for t in uniqueWords]
    max_exp_count = sum(exp_count_array)

    print("Generating distribution")

    # ... (TASK) compute the normalized probabilities of each term.
    # ... using exp_count_array, normalize each value by the total value max_exp_count so that
    # ... they all add up to 1. Store this corresponding array in prob_dist


    # prob_dist = exp_count_array / max_exp_count
    prob_dist = list(map(lambda x: float(x / max_exp_count), exp_count_array))

    print("Filling up sampling table")
    # ... (TASK) create a dict of size table_size where each key is a sequential number and its value is a one-hot index
    # ... the number of sequential keys containing the same one-hot index should be proportional to its prob_dist value
    # ... multiplied by table_size. This table should be stored in cumulative_dict.
    # ... we do this for much faster lookup later on when sampling from this table.

    # cumulative_dict = # ... fill in
    table_size = 1e7

    word_freqs = [int(p * table_size) for p in prob_dist]
    cumulative_dict = {}

    j = 0
    for ind, freq in enumerate(word_freqs):
        i = 0
        while i < freq:
            cumulative_dict[j] = ind
            i += 1
            j += 1

    return cumulative_dict


# .................................................................................
# ... generate a specific number of negative samples
# .................................................................................


def generateSamples(context_idx, num_samples):
    global samplingTable, uniqueWords, randcounter
    results = []
    # ... (TASK) randomly sample num_samples token indices from samplingTable.
    # ... don't allow the chosen token to be context_idx.
    # ... append the x indices to results

    while len(results) < num_samples:
        indice = np.random.randint(low=0, high=len(samplingTable))
        val = samplingTable[indice]
        if context_idx == val:
            continue
        else:
            results.append(val)
    return results


# .................................................................................
# ... learn the weights for the input-hidden and hidden-output matrices
# .................................................................................


@jit(nopython=True)
def performDescent(num_samples, learning_rate, center_token, sequence_chars, W1, W2, negative_indices):
    # sequence chars was generated from the mapped sequence in the core code
    nll_new = 0
    for k in range(0, len(sequence_chars)):
        h = W1[center_token]

        v_j_old_dash_context_word = np.copy(W2[sequence_chars[k]])
        v_j_old_dash_negative_one = np.copy(W2[negative_indices[num_samples * k]])
        v_j_old_dash_negative_two = np.copy(W2[negative_indices[num_samples * k + 1]])

        W2[sequence_chars[k]] = W2[sequence_chars[k]] - learning_rate * (
            sigmoid(np.dot(W2[sequence_chars[k]], h)) - 1) * h
        W2[negative_indices[num_samples * k]] = W2[negative_indices[num_samples * k]] - learning_rate * (
            sigmoid(np.dot(W2[negative_indices[num_samples * k]], h)) - 0) * h
        W2[negative_indices[num_samples * k + 1]] = W2[negative_indices[num_samples * k + 1]] - learning_rate * (
            sigmoid(np.dot(W2[negative_indices[num_samples * k + 1]], h)) - 0) * h

        W1[center_token] = W1[center_token] - learning_rate * (
            (sigmoid(np.dot(v_j_old_dash_context_word, h)) - 1) * v_j_old_dash_context_word + sigmoid(
                np.dot(v_j_old_dash_negative_one, h)) * v_j_old_dash_negative_one +
            sigmoid(np.dot(v_j_old_dash_negative_two, h)) * v_j_old_dash_negative_two)

        nll_first_term = -np.log(sigmoid(np.dot(W2[sequence_chars[k]], W1[center_token])))
        nll_second_term = 0
        for i in range(num_samples):
            nll_second_term += np.log(sigmoid(-np.dot(W2[negative_indices[num_samples * k + i]], W1[center_token])))
        nll_new += nll_first_term - nll_second_term

    return [nll_new]


def trainer(curW1=None, curW2=None):
    global uniqueWords, wordcodes, fullsequence, vocab_size, hidden_size, np_randcounter, randcounter
    vocab_size = len(uniqueWords)  # ... unique characters
    hidden_size = 100  # ... number of hidden neurons
    context_window = [-2, -1, 1, 2]
    nll_results = []  # ... keep array of negative log-likelihood after every 1000 iterations
    iteration_number = []
    # ... determine how much of the full sequence we can use while still accommodating the context window
    start_point = int(math.fabs(min(context_window)))
    end_point = len(fullsequence) - (max(max(context_window), 0))
    mapped_sequence = fullsequence

    # ... initialize the weight matrices. W1 is from input->hidden and W2 is from hidden->output.
    if curW1 == None:
        np_randcounter += 1
        W1 = np.random.uniform(-.5, .5, size=(vocab_size, hidden_size))
        W2 = np.random.uniform(-.5, .5, size=(vocab_size, hidden_size))
    else:
        # ... initialized from pre-loaded file
        W1 = curW1
        W2 = curW2

    # ... set the training parameters
    epochs = 5
    num_samples = 2
    learning_rate = 0.05
    nll = 0
    iternum = 0

    file_handle = open("nll_results_minus_four.csv", "w")
    stri = ""

    # ... Begin actual training
    for j in range(0, epochs):
        print("Epoch: ", j)
        prevmark = 0

        # ... For each epoch, redo the whole sequence...
        for i in range(start_point, end_point):

            if (float(i) / len(mapped_sequence)) >= (prevmark + 0.1):
                print("Progress: ", round(prevmark + 0.1, 1))
                prevmark += 0.1
            if iternum % 10000 == 0:
                print("Negative likelihood: ", nll)
                print("Iteration Number: ", iternum)
                nll_results.append(nll)
                iteration_number.append(iternum)
                nll = 0

            # ... (TASK) determine which token is our current input. Remember that we're looping through mapped_sequence
            if wordcodes["<UNK>"] == mapped_sequence[i]:
                continue
            center_token = mapped_sequence[i]  # ... fill in
            # ... (TASK) don't allow the center_token to be <UNK>. move to next iteration if you found <UNK>.


            iternum += 1
            # ... now propagate to each of the context outputs
            # for k in range(0, len(context_window)):

            mapped_context = [mapped_sequence[i + ctx] for ctx in context_window]
            negative_indices = []
            for q in mapped_context:
                negative_indices += generateSamples(q, num_samples)
            # ... implement gradient descent
            [nll_new] = performDescent(num_samples, learning_rate, center_token, mapped_context, W1, W2,
                                       negative_indices)
            nll += nll_new

    for i in range(len(nll_results)):
        stri += str(nll_results[i]) + "\t" + str(iteration_number[i]) + "\n"
    file_handle.write(stri)

    return [W1, W2]


# .................................................................................
# ... Load in a previously-saved model. Loaded model's hidden and vocab size must match current model.
# .................................................................................

def load_model():
    handle = open("saved_W1_minus_four.data", "rb")
    W1 = np.load(handle)
    handle.close()
    handle = open("saved_W2_minus_four.data", "rb")
    W2 = np.load(handle)
    handle.close()
    return [W1, W2]


# .................................................................................
# ... Save the current results to an output file. Useful when computation is taking a long time.
# .................................................................................

def save_model(W1, W2):
    handle = open("saved_W1_minus_four.data", "wb+")
    np.save(handle, W1, allow_pickle=False)
    handle.close()

    handle = open("saved_W2_minus_four.data", "wb+")
    np.save(handle, W2, allow_pickle=False)
    handle.close()


# ... so in the word2vec network, there are actually TWO weight matrices that we are keeping track of. One of them represents the embedding
# ... of a one-hot vector to a hidden layer lower-dimensional embedding. The second represents the reversal: the weights that help an embedded
# ... vector predict similarity to a context word.






# .................................................................................
# ... code to start up the training function.
# .................................................................................
word_embeddings = []
proj_embeddings = []


def train_vectors(preload=False):
    global word_embeddings, proj_embeddings
    if preload:
        [curW1, curW2] = load_model()
    else:
        curW1 = None
        curW2 = None
    [word_embeddings, proj_embeddings] = trainer(curW1, curW2)
    save_model(word_embeddings, proj_embeddings)


# .................................................................................
# ... find top 10 most similar words to a target word
# .................................................................................

def prediction(target_word):
    # ... (TASK) search through all uniqueWords and for each token, compute its similarity to target_word.
    # ... you will compute this using the absolute cosine similarity of the word_embeddings for the word pairs.
    # ... Note that the cosine() function from scipy.spatial.distance computes a DISTANCE so you need to convert that to a similarity.
    # ... return a list of top 10 most similar words in the form of dicts,
    # ... each dict having format: {"word":<token_name>, "score":<cosine_similarity>}


    global word_embeddings, uniqueWords, wordcodes

    pred = {}

    for uniqueWord in uniqueWords:
        cos_similarity = abs(1 - scipy.spatial.distance.cosine(word_embeddings[wordcodes[uniqueWord]],
                                                               word_embeddings[wordcodes[target_word]]))
        pred[uniqueWord] = cos_similarity

    return dict(Counter(pred).most_common(11))


def calculate_cosine_similarity(word1, word2):
    global word_embeddings, uniqueWords, wordcodes

    cos_similarity = abs(1 - scipy.spatial.distance.cosine(word_embeddings[wordcodes[word1]],
                                                           word_embeddings[wordcodes[word2]]))
    return cos_similarity


def learn_morphology(train_data):
    global word_embeddings, proj_embeddings, uniqueWords, wordcodes
    s_suffix = []
    for d in train_data:
        s_suffix.append(word_embeddings[wordcodes[d[0]]] - word_embeddings[wordcodes[d[1]]])
    return np.mean(s_suffix)


def knearest(target_word_vector, k, morphological_variation):
    global word_embeddings, uniqueWords, wordcodes
    pred = {}
    for uniqueWord in uniqueWords:
        cos_similarity = abs(1 - scipy.spatial.distance.cosine(word_embeddings[wordcodes[uniqueWord]],
                                                               target_word_vector))
        pred[uniqueWord] = cos_similarity

    k_nearest = dict(Counter(pred).most_common(k + 1))
    print(k_nearest)
    i = 0
    for key in k_nearest:
        i += 1
        if morphological_variation == key:
            return i


def get_or_impute_vector(test_data, suffix_vector):
    global word_embeddings, uniqueWords, wordcodes
    precision_k = []
    for d in test_data:
        total_word_vector = word_embeddings[wordcodes[d[1]]] + suffix_vector
        precision_k.append(knearest(total_word_vector, 20, d[0]))
    return precision_k


if __name__ == '__main__':

    if len(sys.argv) == 2:
        filename = sys.argv[1]
        # ... load in the file, tokenize it and assign each token an index.
        # ... the full sequence of characters is encoded in terms of their one-hot positions

        fullsequence = loadData(filename)

        print("Length of Full Sequence: ", len(fullsequence))
        print("Full sequence loaded...")
        # print(uniqueWords)
        print(len(uniqueWords))

        # ... now generate the negative sampling table
        print("Total unique words: ", len(uniqueWords))
        print("Preparing negative sampling table")
        samplingTable = negativeSampleTable(fullsequence, uniqueWords, wordcounts)

        train_vectors(preload=False)
        [word_embeddings, proj_embeddings] = load_model()

        # TASK 2 START
        '''
        s_suffix = [['years', 'year'],
                    ['friends', 'friend'],
                    ['circumstances', 'circumstance'],
                    ['kids', 'kid'],
                    ['hours', 'hour'],
                    ['films', 'film'],
                    ['movies', 'movies'],
                    ['characters', 'character'],
                    ['places', 'place'],
                    ['angles', 'angle'],
                    ['objects', 'object'],
                    ['colors', 'color'],
                    ['actors', 'actor'],
                    ['tickets', 'ticket'],
                    ['epics', 'epic'],
                    ['flashbacks', 'flashback'],
                    ['talents', 'talent'],
                    ['thousands', 'thousand'],
                    ['directors', 'director'],
                    ['horses', 'horse'],
                    ["types", "type"],
                    ["ships", "ship"],
                    ["values", "value"],
                    ["walls", "wall"],
                    ["spoilers", "spoiler"]]

        train_s = s_suffix[:-2]
        test_s = s_suffix[22:]

        s_suffix = learn_morphology(train_s)
        precision_s = get_or_impute_vector(test_s, s_suffix)

        # Prints precision@k for all test_s words and the average
        print(precision_s)
        print(np.mean(precision_s))

        ed_suffix = [["wanted", "want"],
                     ["needed", "need"],
                     ["walked", "walk"],
                     ["talented", "talent"],
                     ["informed", "inform"],
                     ["called", "call"],
                     ["selected", "select"],
                     ["repeated", "repeat"],
                     ["edited", "edit"],
                     ["witnessed", "witness"],
                     ["pulled", "pull"],
                     ["finished", "finish"],
                     ["started", "start"],
                     ["seemed", "seem"],
                     ["reminded", "remind"],
                     ["displayed", "display"],
                     ["happened", "happen"],
                     ["laughed", "laugh"],
                     ["asked", "ask"],
                     ["played", "play"]
                     ]

        train_ed = ed_suffix[:-2]
        test_ed = ed_suffix[18:]

        ed_suffix = learn_morphology(train_ed)
        precision_ed = get_or_impute_vector(test_ed, ed_suffix)

        # Prints precision@k for all test_ed words and the average
        print(precision_ed)
        print(np.mean(precision_ed))
        '''

        # TASK 2 END

        # TASK 3 START
        '''
        targets = ["bad", "good", "scary", "funny"]
        file_handle = open("p9_output.txt", "w")
        stri = "target_word,similar_word,similar_score" + "\n"
        for targ in targets:
            bestpreds = prediction(targ)
            for word in bestpreds:
                stri += targ + "," + word + "," + str(bestpreds[word]) + "\n"
        file_handle.write(stri)

        '''
        # TASK 3 END





        # TASK 4 START

        '''
        reader = open("extrinsic-train.tsv", "r")
        header = next(reader)
        train_data = []
        train_target = []
        for row in reader:
            stop_words = set(stopwords.words('english'))
            arr = row[:-1].split('\t')
            # word_tokens = better_tokenize(arr[2])
            word_tokens = nltk.word_tokenize(arr[2])
            words = [w for w in word_tokens if not w in stop_words]
            final_sum = np.zeros(word_embeddings.shape[1])
            count_words = 0
            for word in words:
                if word in uniqueWords:
                    count_words += 1
                    final_sum += word_embeddings[wordcodes[word]]

            final_sum = final_sum / count_words

            train_data.append(final_sum)
            train_target.append(arr[1])

        reader = open("extrinsic-dev.tsv", "r")
        header = next(reader)
        dev_data = []
        dev_target = []
        for row in reader:
            stop_words = set(stopwords.words('english'))
            arr = row[:-1].split('\t')
            # word_tokens = better_tokenize(arr[2])
            word_tokens = nltk.word_tokenize(arr[2])
            words = [w for w in word_tokens if not w in stop_words]
            final_sum = np.zeros(word_embeddings.shape[1])
            count_words = 0
            for word in words:
                if word in uniqueWords:
                    count_words += 1
                    final_sum += word_embeddings[wordcodes[word]]

            final_sum = final_sum / count_words

            dev_data.append(final_sum)
            dev_target.append(arr[1])

        clf = LogisticRegression(C=100.0).fit(train_data, train_target)

        print('Accuracy of Logistic regression classifier on training set: {:.2f}'
              .format(clf.score(train_data, train_target)))
        print('Accuracy of Logistic regression classifier on dev set: {:.2f}'
              .format(clf.score(dev_data, dev_target)))

        reader = open("extrinsic-test.tsv", "r")
        header = next(reader)
        test_data = []

        file_handle = open("results.txt", "w")
        output_str = ""

        for row in reader:
            stop_words = set(stopwords.words('english'))
            arr = row[:-1].split('\t')
            word_tokens = nltk.word_tokenize(arr[1])
            words = [w for w in word_tokens if not w in stop_words]
            final_sum = np.zeros(word_embeddings.shape[1])
            count_words = 0
            for word in words:
                if word in uniqueWords:
                    count_words += 1
                    final_sum += word_embeddings[wordcodes[word]]

            final_sum = final_sum / count_words

            output_str += arr[0] + "\t" + str(clf.predict([final_sum])[0]) + "\n"
        file_handle.write(output_str)
        '''
        # TASK 4 END

    else:
        print("Please provide a valid input filename")
        sys.exit()
