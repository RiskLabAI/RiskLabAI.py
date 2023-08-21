# from collections import Counter
# from math import log2
# from typing import Dict, Tuple


# def shannon_entropy(message: str) -> float:
#     """
#     Calculate Shannon Entropy.

#     :param message: Input encoded message
#     :type message: str
#     :return: Calculated Shannon Entropy
#     :rtype: float
#     """
#     character_counts = Counter(message)
#     message_length = len(message)

#     entropy = -sum(
#         count / message_length * log2(count / message_length)
#         for count in character_counts.values()
#     )

#     return entropy


# def lempel_ziv_entropy(message: str) -> float:
#     """
#     Calculate Lempel-Ziv Entropy.

#     :param message: Input encoded message
#     :type message: str
#     :return: Calculated Lempel-Ziv Entropy
#     :rtype: float
#     """
#     library = set()
#     message_length = len(message)
#     i = 0

#     while i < message_length:
#         j = i
#         while message[i:j + 1] in library and j < message_length:
#             j += 1
#         library.add(message[i:j + 1])
#         i = j + 1

#     return len(library) / message_length


# def probability_mass_function(
#     message: str,
#     approximate_word_length: int
# ) -> Dict[str, float]:
#     """
#     Calculate Probability Mass Function.

#     :param message: Input encoded message
#     :type message: str
#     :param approximate_word_length: Approximation of word length
#     :type approximate_word_length: int
#     :return: Probability Mass Function
#     :rtype: dict
#     """
#     library = Counter(message[i:i + approximate_word_length]
#                      for i in range(len(message) - approximate_word_length + 1))

#     denominator = float(len(message) - approximate_word_length)
#     probability_mass_function_ = {key: len(library[key]) / denominator
#                                  for key in library}

#     return probability_mass_function_


# def plug_in_entropy_estimator(
#     message: str,
#     approximate_word_length: int = 1
# ) -> float:
#     """
#     Calculate Plug-in Entropy Estimator.

#     :param message: Input encoded message
#     :type message: str
#     :param approximate_word_length: Approximation of word length, default is 1
#     :type approximate_word_length: int
#     :return: Calculated Plug-in Entropy Estimator
#     :rtype: float
#     """
#     pmf = probability_mass_function(message, approximate_word_length)
#     plug_in_entropy_estimator = -sum(
#         pmf[key] * log2(pmf[key])
#         for key in pmf.keys()
#     ) / approximate_word_length

#     return plug_in_entropy_estimator


# def longest_match_length(
#     message: str,
#     i: int,
#     n: int
# ) -> Tuple[int, str]:
#     """
#     Calculate the length of the longest match.

#     :param message: Input encoded message
#     :type message: str
#     :param i: Index value
#     :type i: int
#     :param n: Length parameter
#     :type n: int
#     :return: Tuple containing matched length and substring
#     :rtype: tuple
#     """
#     longest_match = ""
#     for l in range(1, n + 1):
#         pattern = message[i:i + l + 1]
#         for j in range(i - n + 1, i + 1):
#             candidate = message[j:j + l + 1]
#             if pattern == candidate:
#                 longest_match = pattern
#                 break

#     return len(longest_match) + 1, longest_match


# def kontoyiannis_entropy(
#     message: str,
#     window: int = None
# ) -> float:
#     """
#     Calculate Kontoyiannis Entropy.

#     :param message: Input encoded message
#     :type message: str
#     :param window: Length of expanding window, default is None
#     :type window: int or None
#     :return: Calculated Kontoyiannis Entropy
#     :rtype: float
#     """
#     output = {"num": 0, "sum": 0, "sub_string": []}
#     message_length = len(message)

#     if window is None:
#         points = range(2, message_length // 2 + 2)
#     else:
#         window = min(window, message_length // 2)
#         points = range(window + 1, message_length - window + 2)

#     for i in points:
#         n = i if window is None else window
#         l, sub_string = longest_match_length(message, i, n)
#         output["sum"] += log2(n) / l
#         output["sub_string"].append(sub_string)
#         output["num"] += 1

#     output["h"] = output["sum"] / output["num"]
#     output["r"] = 1 - output["h"] / log2(message_length)

#     return output["h"]
