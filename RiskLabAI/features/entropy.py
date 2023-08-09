from math import log2

def shannon_entropy(message: str) -> float:
    """
    Calculate Shannon Entropy.

    :param message: Input encoded message
    :type message: str
    :return: Calculated Shannon Entropy
    :rtype: float
    """
    char_to_count = {}
    entropy = 0

    for character in message:
        char_to_count[character] = char_to_count.get(character, 0) + 1

    message_length = len(message)
    for count in char_to_count.values():
        frequency = count / message_length
        entropy -= frequency * log2(frequency)

    return entropy

def lemple_ziv_entropy(message: str) -> float:
    """
    Calculate Lemple-Ziv Entropy.

    :param message: Input encoded message
    :type message: str
    :return: Calculated Lemple-Ziv Entropy
    :rtype: float
    """
    i, library = 0, set([str(message[1])])
    message_length = len(message)

    while i < message_length:
        last_j_value = message_length - 1

        for j in range(i, message_length):
            message_ = message[i + 1:j + 2]

            if message_ not in library:
                library.add(message_)
                last_j_value = j
                break

        i = last_j_value + 1

    return len(library) / len(message)

def probability_mass_function(message: str, approximate_word_length: int) -> dict:
    """
    Calculate Probability Mass Function.

    :param message: Input encoded message
    :type message: str
    :param approximate_word_length: Approximation of word length
    :type approximate_word_length: int
    :return: Probability Mass Function
    :rtype: dict
    """
    library = {}

    message_length = len(message)
    for index in range(approximate_word_length, message_length):
        message_ = message[index - approximate_word_length:index]

        if message_ not in library:
            library[message_] = [index - approximate_word_length]
        else:
            library[message_].append(index - approximate_word_length)

    denominator = float(message_length - approximate_word_length)
    probability_mass_function_ = {
        key: len(library[key]) / denominator for key in library
    }

    return probability_mass_function_

def plug_in_entropy_estimator(message: str, approximate_word_length: int = 1) -> float:
    """
    Calculate Plug-in Entropy Estimator.

    :param message: Input encoded message
    :type message: str
    :param approximate_word_length: Approximation of word length, default is 1
    :type approximate_word_length: int
    :return: Calculated Plug-in Entropy Estimator
    :rtype: float
    """
    pmf = probability_mass_function(message, approximate_word_length)
    plug_in_entropy_estimator = -sum([pmf[key] * log2(pmf[key]) for key in pmf.keys()]) / approximate_word_length
    return plug_in_entropy_estimator

def longest_match_length(message: str, i: int, n: int) -> tuple:
    """
    Calculate the length of the longest match.

    :param message: Input encoded message
    :type message: str
    :param i: Index value
    :type i: int
    :param n: Length parameter
    :type n: int
    :return: Tuple containing matched length and substring
    :rtype: tuple
    """
    sub_string = ""
    for l in range(1, n + 1):
        message1 = message[i:i + l + 1]
        for j in range(i - n + 1, i + 1):
            message0 = message[j:j + l + 1]
            if message1 == message0:
                sub_string = message1
                break

    return len(sub_string) + 1, sub_string

def kontoyiannis_entorpy(message: str, window: int = 0) -> float:
    """
    Calculate Kontoyiannis Entropy.

    :param message: Input encoded message
    :type message: str
    :param window: Length of expanding window, default is 0
    :type window: int
    :return: Calculated Kontoyiannis Entropy
    :rtype: float
    """
    output = {"num": 0, "sum": 0, "subString": []}

    if window is None:
        points = range(2, len(message) // 2 + 2)
    else:
        window = min(window, len(message) // 2)
        points = range(window + 1, len(message) - window + 2)

    for i in points:
        if window is None:
            l, message_ = longest_match_length(message, i, i)
            output["sum"] += log2(i) / l
        else:
            l, message_ = longest_match_length(message, i, window)
            output["sum"] += log2(window) / l

        output["subString"].append(message_)
        output["num"] += 1

    output["h"] = output["sum"] / output["num"]
    output["r"] = 1 - output["h"] / log2(len(message))

    return output["h"]
