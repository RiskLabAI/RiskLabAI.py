from math import log2

"""
function: Shannon Entropy 
reference: De Prado, M. (18) Advances in Financial Machine Learning
methodology: page 263 SHANNON’S ENTROPY section
"""
def shannon_entropy(
    message:str # input encoded message
) -> float:

    char_to_count = dict()
    entropy = 0
    for character in message:
        try:
            char_to_count[character] += 1
        except:
            char_to_count[character] = 1

    message_length = len(message)
    for count in char_to_count.values():
        frequency = count / message_length
        entropy -= frequency * log2(frequency)

    return entropy


"""
function: A LIBRARY BUILT USING THE LZ ALGORITHM
reference: De Prado, M. (18) Advances in Financial Machine Learning
methodology: page 266 LEMPEL-ZIV ESTIMATORS section
"""
def lemple_ziv_entropy(
    message:str # input encoded message
) -> float:

    i, library = 0, set([str(message[1])])
    message_length = len(message)
    while i < message_length:
        last_j_value = message_length - 1
        for j in range(i, message_length):
            message_ = message[i+1:j+2]
            if message_ not in library:
                library.add(message_)
                last_j_value = j
                break

        i = last_j_value + 1


    return len(library) / len(message)


from math import log2
def probability_mass_function(
    message:str, # input encoded message
    approximate_word_length:int # approximation of word length
) -> dict:

    library = dict()

    message_length = len(message)
    for index in range(approximate_word_length, message_length):
        message_ = message[index-approximate_word_length:index] 

        if message_ not in library:
            library[message_] = [index-approximate_word_length]
        else:
            library[message_] = library[message_] + [index-approximate_word_length]
            # library[message_].append(index-approximate_word_length)


    denominator = float(message_length - approximate_word_length)
    probability_mass_function_ = {
        key: len(library[key]) / denominator for key in library
    }

    return probability_mass_function_

"""
function: Plug-in Entropy Estimator Implementation
reference: De Prado, M. (18) Advances in Financial Machine Learning
methodology: page 265 THE PLUG-IN (OR MAXIMUM LIKELIHOOD) ESTIMATOR section
"""
def plug_in_entropy_estimator(
    message:str, # input encoded message
    approximate_word_length:int=1, # approximation of word length
) -> float:

    pmf = probability_mass_function(message, approximate_word_length)
    plugInEntropyEstimator = -sum([pmf[key] * log2(pmf[key]) for key in pmf.keys()]) / approximate_word_length
    return plugInEntropyEstimator
    


"""
function: COMPUTES THE LENGTH OF THE LONGEST MATCH
reference: De Prado, M. (18) Advances in Financial Machine Learning
methodology: page 267 LEMPEL-ZIV ESTIMATORS section
"""
def longest_match_length(
    message:str,
    i:int,
    n:int
) -> tuple:

    sub_string = ""
    for l in range(1, n + 1):
        message1 = message[i:i+l+1]
        for j in range(i-n+1, i+1):
            message0 = message[j:j+l+1]
            if message1 == message0:
                sub_string = message1
                break # search for higher l.


    return (len(sub_string) + 1, sub_string) # matched length + 1 


"""
function: IMPLEMENTATION OF ALGORITHMS DISCUSSED IN GAO ET AL.
reference: De Prado, M. (18) Advances in Financial Machine Learning
methodology: page 268 LEMPEL-ZIV ESTIMATORS section
"""
def kontoyiannis_entorpy(
    message:str, # input encoded message
    window:int=0, # length of expanding window 
) -> float:

    output = {"num" : 0, "sum" : 0, "subString" : []}
    if window == None:
        points = range(2, len(message)/2+2)
    else:
        window = min(window, len(message) / 2)
        points = range(window+1, len(message)-window+2)

    for i in points:
        if window == None:
            (l, message_) = longest_match_length(message, i, i)
            output["sum"] += log2(i) / l # to avoid Doeblin condition
        else:
            (l, message_) = longest_match_length(message, i, window)
            output["sum"] += log2(window) / l # to avoid Doeblin condition

        output["subString"].append(message_)
        output["num"] += 1

    output["h"] = output["sum"] / output["num"]
    output["r"] = 1 - output["h"] / log2(len(message)) 

    return output["h"] 

