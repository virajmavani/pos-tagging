import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from nltk.corpus import brown
from collections import Counter


rare_words = set()
feature_dict = {}
tag_dict = {}


def word_ngram_features(i, words):
    """

    :param i:
    :param words:
    :return:
    """
    copy_ = words.copy()
    copy_.insert(0, "<s>")
    copy_.insert(0, "<s>")
    copy_.append("</s>")
    copy_.append("</s>")
    # Offset of 2 due to padding
    i += 2
    features = list()
    features.append("prevbigram-" + copy_[i-1])
    features.append("nextbigram-" + copy_[i+1])
    features.append("prevskip-" + copy_[i-2])
    features.append("nextskip-" + copy_[i+2])
    features.append("prevtrigram-" + copy_[i-1] + "-" + copy_[i-2])
    features.append("nexttrigram-" + copy_[i+1] + "-" + copy_[i+2])
    features.append("centertrigram-" + copy_[i-1] + "-" + copy_[i+1])
    return features


def word_features(word, rare_words):
    """

    :param word:
    :param rare_words:
    :return:
    """
    features = list()
    if word not in rare_words:
        features.append("word-" + word)
    if word[0].isupper():
        features.append("capital")
    if any(char.isdigit() for char in word):
        features.append("number")
    if any(char == "-" for char in word):
        features.append("hyphen")
    for j in range(1, 5):
        if j <= len(word):
            features.append("prefix" + str(j) + "-" + word[:j])
    for j in range(len(word)-1, len(word)-5, -1):
        if j >= 0:
            features.append("suffix" + str(len(word)-j) + "-" + word[j:])
    return features


def get_features(i, words, prevtag, rare_words):
    """

    :param i:
    :param words:
    :param prevtag:
    :param rare_words:
    :return:
    """
    features = word_ngram_features(i, words)
    features.extend(word_features(words[i], rare_words))
    features.append("tagbigram-" + prevtag)
    features = [feature.lower() for feature in features]
    return features


def remove_rare_features(features, n):
    freq = Counter(feature for sentence in features for word in sentence for feature in word)
    rare_features = set()
    non_rare_features = set()
    for feature in freq.keys():
        if freq[feature] < n:
            rare_features.add(feature)
        else:
            non_rare_features.add(feature)
    new_features = [[[feature for feature in word if feature not in rare_features] for word in sentence] for sentence in features]
    return new_features


def build_Y(tags):
    Y = [tag_dict[tag] for sentence in tags for tag in sentence]
    return np.array(Y)


def build_X(features):
    examples = []
    features_ = []
    i = 0
    for sentence in features:
        for feature_list in sentence:
            for feature in feature_list:
                if feature_dict.get(feature, None):
                    features_.append(feature_dict[feature])
                else:
                    continue
                examples.append(i)
            i += 1
    values = [1]*len(examples)
    examples = np.array(examples)
    features_ = np.array(features_)
    values = np.array(values)
    return csr_matrix((values, (examples, features_)), shape=(i, len(feature_dict)))


def load_test(filename):
    test_file = open(filename)
    test_set = test_file.readlines()
    test_set = [[word.strip() for word in sentence.split()] for sentence in test_set]
    return test_set


def get_predictions(test_sentence, model):
    Y_pred = np.empty([len(test_sentence)-1, len(tag_dict), len(tag_dict)])
    for i in range(1, len(test_sentence)):
        for tag, index in tag_dict.items():
            features = get_features(i, test_sentence, tag, rare_words)
            features = [[features]]
            X = build_X(features)
            Y_pred[i-1][index] = model.predict_log_proba(X)
    features_start = get_features(0, test_sentence, "<S>", rare_words)
    features_start = [[features_start]]
    X_start = build_X(features_start)
    Y_start = model.predict_log_proba(X_start)
    return Y_pred, Y_start


def viterbi(Y_start, Y_pred):
    V = np.empty(shape=(Y_pred.shape[0]+1, len(tag_dict)))
    BP = np.empty(shape=(Y_pred.shape[0]+1, len(tag_dict)))
    for j in range(np.shape(Y_pred)[1]):
        V[0, j] = Y_start[0, j]
        BP[0, j] = -1
    for i in range(np.shape(Y_pred)[0]):
        for tag, k in tag_dict.items():
            num_arr = np.empty([1, len(tag_dict)])
            for _, j in tag_dict.items():
                num_arr[0, j] = V[i, j] + Y_pred[i, j, k]
            V[i + 1, k] = num_arr.max()
            BP[i + 1, k] = np.argmax(num_arr)

    backward_indices = []

    n = np.shape(V)[0] - 1
    index = np.argmax(V[n])
    backward_indices.append(index)

    while (n >= 1):
        index = int(BP[n, index])
        backward_indices.append(index)
        n = n - 1

    backward_indices = backward_indices[::-1]
    tag_list = []
    for index in backward_indices:
        for tag, i in tag_dict.items():
            if i == index:
                tag_list.append(tag)
    return tag_list


def main():
    brown_sentences = brown.tagged_sents(tagset='universal')
    train_sentences = [[word for word, tag in sentence] for sentence in brown_sentences]
    train_tags = [[tag for word, tag in sentence] for sentence in brown_sentences]
    freq = Counter(word for sentence in train_sentences for word in sentence)

    rare_words.update([word for word in freq.keys() if (freq[word] < 5)])
    training_features = list()
    for i in range(len(train_sentences)):
        sentence = train_sentences[i]
        training_features.append([])
        prevtag = "<S>"
        for j in range(len(sentence)):
            training_features[i].append(get_features(j, sentence, prevtag, rare_words))
            prevtag = train_tags[i][j]
    new_features = remove_rare_features(training_features, 5)

    idx = 0
    for sentence in new_features:
        for word in sentence:
            for feature in word:
                if feature_dict.get(feature, None) is None:
                    feature_dict[feature] = idx
                    idx += 1

    idx = 0
    for sentence in train_tags:
        for tag in sentence:
            if tag_dict.get(tag, None) is None:
                tag_dict[tag] = idx
                idx += 1

    X_train = build_X(new_features)
    Y_train = build_Y(train_tags)

    print(X_train.shape)
    print(Y_train.shape)

    model = LogisticRegression(class_weight='balanced', solver='saga', multi_class='multinomial')
    model.fit(X_train, Y_train)

    test_sentences = load_test("test.txt")
    print(test_sentences)
    for sentence in test_sentences:
        Y_pred, Y_start = get_predictions(sentence, model)
        tag_sequence = viterbi(Y_start, Y_pred)
        print(tag_sequence)


if __name__ == '__main__':
    main()
