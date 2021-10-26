import string
import unicodedata
import numpy as np
import json
import itertools
from collections import Counter
import pandas as pd
import pint
import matplotlib.pyplot as plt

# import en_core_web_lg
from scipy.special import logsumexp

LOG_EPS = -1e150
PI = [0.98, 0.01, 0.01]
TYPE_INDEX = 0
MISSING_INDEX = 1
ANOMALIES_INDEX = 2
LLHOOD_TYPE_START_INDEX = 2

# from gensim.models import FastText #KeyedVectors,
# fast_text_model = FastText.load_fasttext_format('/Users/tceritli/Workspace/git/github/units/wiki.en/wiki.en.bin')
# nlp = en_core_web_lg.load()

# KeyedVectors and spacy do not handle out-of vocab
# fast_text_model = KeyedVectors.load_word2vec_format('/Users/tceritli/Workspace/git/github/units/GoogleNews-vectors-negative300.bin', binary=True)


def log_sum_exp(x):
    return logsumexp(x)


def remove_digits(s):
    return "".join(i for i in s if not i.isdigit())


# numeric_const_pattern = r"""
#     [-+]? # optional sign
#      (?:
#          (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
#          |
#          (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
#      )
#      # followed by optional exponent part if desired
#      (?: [Ee] [+-]? \d+ ) ?
#      """
#
# numeric_const_pattern = r"""
#     [-+]? # optional sign
#      (?:
#          (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
#          |
#          (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
#      )
#      # followed by optional exponent part if desired
#      (?: [Ee] [+-]? \d+ ) ?
#      """
# rx = re.compile(numeric_const_pattern, re.VERBOSE)
#
# numeric_with_string_const_pattern = r"""
#     [-+]? # optional sign
#      (?:
#          (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
#          |
#          (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
#      )
#      # followed by optional exponent part if desired
#      (?: [Ee] [+-]? \d+ ) ?
#      (?: \s+ ) ?
#      (?: \S+ ) ?
#      """
#
# numeric_with_string_const_pattern = r"""
#     [-+]? # optional sign
#      (?:
#          (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
#          |
#          (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
#      )
#      # followed by optional exponent part if desired
#      (?: [Ee] [+-]? \d+ ) ?
#
#      # followed by optional any character
#      (?: .*) ?
#      """
#
# numeric_with_string_const_pattern = r"""
#     [-+]? # optional sign
#      (
#          (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
#          |
#          (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
#      )
#
#      # followed by optional any character
#      (.*) ?
#      """
# numeric_with_string_const_pattern = r"""
#     (
#     [-+]? # optional sign
#      (
#          (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
#          |
#          (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
#      ) ?
#      # whitespace as separator
#      (?: [\s]*) ?
#
#      # followed by optional characters (alphanumeric, whitespace, some punctuation marks
#      ([\w\s.!?\\-]*) ?
#      )
#      |
#      (
#      (\w*)?
#
#      # whitespace as separator
#      (?: [\s]*) ?
#
#      [-+]? # optional sign
#      (
#          (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
#          |
#          (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
#      ) ?
#      )
#      """
import re

numeric_with_string_const_pattern = r"""
    [-+]? # optional sign
     (
         (?: \d+ \/ \d+ )   # 1/4 etc
         |
         (?: \d* [.,] \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
         |
         (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc                  
     ) ?
     # whitespace as separator
     (?: [\s]*) ?
     
     # followed by optional characters (alphanumeric, whitespace, some punctuation marks
     ([$£\w\s.!?\\-]*) ?     
     """
rx = re.compile(numeric_with_string_const_pattern, re.VERBOSE)


def parse_cell_value(s):
    # remove leading and trailing whitespace
    s = s.strip()
    # remove trailing dots
    s = s.rstrip(".,")
    # check regular expression
    res = rx.match(s)
    if res is None:
        return ["", ""]
    else:
        magnitude = res.groups()[0]
        unit_symbol = res.groups()[1]
        if magnitude is None:
            magnitude = ""
        magnitude = magnitude.replace(",", "")
        return [magnitude, unit_symbol]


def get_num(s):
    temp = rx.findall(s)
    if len(temp) != 0:
        return temp[0]
    else:
        return ""


float_reg_exp = (
    "[\-+]?(((\d+(\.\d*)?)|\.\d+)([eE][\-+]?[0-9]+)?)|(\d{1,3}(,[0-9]{3})+(\.\d*)?)"
)
any_char_reg_exp = ".*"

float_reg_exp + any_char_reg_exp


def get_unit(s):
    s = s.replace(str(get_num(s)), "")
    return remove_whitespaces_head_and_tail(s)


def remove_whitespaces_head_and_tail(s):
    if len(s) != 0:
        # remove leading and trailing whitespace
        while len(s) != 0 and s[0] == " ":
            s = s[1:]
        while len(s) != 0 and s[-1] == " ":
            s = s[:-1]
    return s


def string_normalisation(s):
    """generates a key from a given string variable 's'
    input params: s
    returns: key"""
    if len(s) != 0:
        # remove leading and trailing whitespace
        while s[0] == " ":
            s = s[1:]
            if len(s) == 0:
                break

        while s[-1] == " ":
            s = s[:-1]
            if len(s) == 0:
                break

        # change all characters to their lowercase representation
        s = s.lower()

        # remove all punctuation and control characters
        translator = str.maketrans("", "", string.punctuation)
        s = s.translate(translator)
        mpa = dict.fromkeys(range(32))
        s = s.translate(mpa)

        # normalize extended western characters to their ASCII representation (for example "gödel" → "godel")
        s = unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("utf-8")

        # remove leading and trailing whitespace
        if len(s) != 0:
            while s[0] == " ":
                s = s[1:]
                if len(s) == 0:
                    break
            while s[-1] == " ":
                s = s[:-1]
                if len(s) == 0:
                    break

        return s
    else:
        return "empty string is given!"


def contains_all(_str, _list):
    res = True
    for s in _str:
        if s not in _list:
            res = False
            break
    return res


def substring_finder(string1, string2, size=6):
    answer = ""
    len1, len2 = len(string1), len(string2)
    if len1 < len2:
        tmp = string1
        string1 = string2
        string2 = tmp
        len1, len2 = len(string1), len(string2)

    for i in range(len1):
        match = ""
        for j in range(len2):
            if i + j < len1 and string1[i + j] == string2[j]:
                match = match + string2[j]
                if (len(match) == size) or (len(match) == min(len1, len2)):
                    answer = match
    return answer


def gen_blocks_all(strings, blocking_size=6):
    N = len(strings)
    blocks = [n for n in range(N)]

    for n in range(N):
        for j in range(N - 1):

            current_string = strings[n]
            next_string = strings[j]
            common_substring = substring_finder(
                current_string, next_string, blocking_size
            )

            if common_substring != "":
                if blocks[n] <= blocks[j]:
                    blocks[j] = blocks[n]
                else:
                    blocks[n] = blocks[j]

    return blocks


def gaussian_likelihood(x, m, std):
    return (1.0 / (2.0 * np.pi * std ** 2)) * (
        np.exp(-1.0 * (x - m) ** 2 / (2 * std ** 2))
    )


def search_units(query, units):
    for unit in units:
        if query in unit["surfaces"] + unit["symbols"] + [
            unit["name"],
        ]:
            return unit["URI"]


def get_units_of_entity(entity, units):
    results = []
    for unit in units:
        if entity == unit["entity"]:
            results.append(unit)
    return results


def get_symbols_for_entity(entity, units):
    results = [unit["symbols"] for unit in units if unit["entity"] == entity]
    return list(itertools.chain(*results))


def get_names_for_symbol(unit_symbol, measurement_type, quantulum_units):
    # return [unit['name'] for unit in units if symbol in unit['symbols']]
    return [
        quantulum_unit["name"]
        for quantulum_unit in quantulum_units
        if "symbols" in quantulum_unit
        and unit_symbol in quantulum_unit["symbols"]
        and measurement_type in quantulum_unit["entity"]
    ]


# def get_names_for_symbol(symbol, units_knowledge):
#     return [units_knowledge[entity_id][1] for entity_id in units_knowledge if symbol in units_knowledge[entity_id][2]]


def get_all_entities(units):
    return [unit["entity"] for unit in units if len(unit["entity"]) != 0]


# def get_all_entities(units_knowledge):
#     return [units_knowledge[entity_id][0] for entity_id in units_knowledge if len(units_knowledge[entity_id][0]) !=0]


def get_all_symbols(units):
    results = [unit["symbols"] for unit in units]
    return list(itertools.chain(*results))


def check_header_for_entities(entities, header):
    return [entity for entity in entities if entity in header]


def check_header_for_symbols(symbols, header):
    return [symbol for symbol in symbols if symbol in header]


def extract_units(raw_values):
    unique_raw_values = np.unique(raw_values)

    # unique_val_to_index[val] = indices where for each index in indices we have raw_values[index] = val
    unique_val_to_index = {
        val: np.where(raw_values == val)[0] for val in unique_raw_values
    }

    # extract units
    unique_raw_str = [remove_digits(val) for val in unique_raw_values]

    nonempty_unique_raw_str = [
        val_str for val_str in unique_raw_str if len(val_str) != 0
    ]
    print("unnormalized units:\n\t", np.unique(nonempty_unique_raw_str), "\n")

    # remove whitespace
    unique_raw_str = [
        remove_whitespaces_head_and_tail(val_str) for val_str in unique_raw_str
    ]

    # maps units to values
    unit_to_val = {i: [] for i in list(np.unique(unique_raw_str))}
    for i in range(len(unique_raw_str)):
        unit_to_val[unique_raw_str[i]].append(list(unique_val_to_index.keys())[i])

    nonempty_unique_raw_str = [
        val_str for val_str in unique_raw_str if len(val_str) != 0
    ]
    fingerprints = [
        string_normalisation(val)
        for val in nonempty_unique_raw_str
        if len(string_normalisation(val)) != 0
    ]
    unique_fingerprints = list(set(fingerprints))
    print("normalized units':\n", unique_fingerprints)

    return nonempty_unique_raw_str, fingerprints, unique_fingerprints, unit_to_val


def clean_units(raw_values):
    unique_raw_values = np.unique(raw_values)
    res = {
        unique_raw_value: {"unit": "", "value": ""}
        for unique_raw_value in unique_raw_values
    }

    # extract units
    unnormalized_units = []
    for unique_raw_value in unique_raw_values:
        num = get_num(unique_raw_value)
        if len(str(num)) != 0:
            res[unique_raw_value]["value"] = num

        unit = remove_digits(unique_raw_value)
        if unit not in unnormalized_units:
            unnormalized_units.append(unit)
        if len(unit) != 0:
            normalized_unit = string_normalisation(unit)

            if len(normalized_unit) != 0:
                res[unique_raw_value]["unit"] = normalized_unit
    print(unnormalized_units)
    return res


def create_units_to_val(fingerprints, nonempty_unique_raw_str, unit_to_val):
    units_to_val = {}
    for i, (val, nonempty_unique_raw_str_val) in enumerate(
        zip(fingerprints, nonempty_unique_raw_str)
    ):
        if val in units_to_val:
            units_to_val[val] = (
                units_to_val[val] + unit_to_val[nonempty_unique_raw_str_val].copy()
            )
        else:
            units_to_val[val] = unit_to_val[nonempty_unique_raw_str_val].copy()
    return units_to_val


def create_data_values(units_to_val):
    data_values = {}
    for key in units_to_val:
        nums = [get_num(val) for val in units_to_val[key] if get_num(val) != ""]
        if len(nums) != 0:
            data_values[key] = nums
    return data_values


def plot_data_values(data_values):
    plt.figure(figsize=(15, 5))
    for key in data_values:
        plt.scatter(
            [key for i in range(len(data_values[key]))], data_values[key], label=key
        )
    plt.xticks(rotation="vertical")
    plt.xlabel("units")
    plt.ylabel("data values")
    plt.show()


def plot_data_values_together(data_values, data_values_converted):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))

    for key in data_values:
        axes[0].scatter(
            [key for i in range(len(data_values[key]))], data_values[key], label=key
        )
        axes[0].xaxis.set_tick_params(rotation=90)
        axes[0].set_xlabel("units")
        axes[0].set_ylabel("data values")
        axes[0].set_title("before conversion")

    for key in data_values_converted:
        axes[1].scatter(
            [key for i in range(len(data_values_converted[key]))],
            data_values_converted[key],
            label=key,
        )
        axes[1].xaxis.set_tick_params(rotation=90)
        axes[1].set_xlabel("units")
        axes[1].set_ylabel("data values")
        axes[1].set_title("after conversion")
    plt.show()


def get_data_values(units_dict, res):
    data_values = {unit: [] for unit in units_dict}
    for element in res:
        data_values[res[element]["unit"]].append(res[element]["value"])
    return data_values


def use_ontology(units_dict, units):
    unit_names = {}
    for normalized_unit in units_dict:
        names = get_names_for_symbol(normalized_unit, units)
        if len(names) != 0:
            unit_names[normalized_unit] = names[0]
    return unit_names


def normalize_results(res, column_unit, unit_names):
    res_converted = res.copy()
    for element in res_converted:
        if res_converted[element]["unit"] != column_unit:
            res_converted[element]["value"] = convert_value(
                res_converted[element]["value"],
                unit_names[res_converted[element]["unit"]],
                column_unit,
            ).magnitude
            res_converted[element]["unit"] = (
                res_converted[element]["unit"] + "(in " + column_unit + ")"
            )
    return res_converted


def get_normalized_data_values(res_converted):
    data_values_converted = {}
    for element in res_converted:
        element_unit = res_converted[element]["unit"]
        if element_unit in data_values_converted:
            data_values_converted[element_unit].append(res_converted[element]["value"])
        else:
            data_values_converted[element_unit] = [
                res_converted[element]["value"],
            ]
    return data_values_converted


def get_raw_values(filepath, column_name):
    df = pd.read_csv(filepath, sep=",", encoding="ISO-8859-1", keep_default_na=False)
    df[column_name].replace("", np.nan, inplace=True)
    df_col = df[column_name].dropna(axis=0, how="any")
    return df_col.values


def convert_value(_value, _from, _to):
    ureg = pint.UnitRegistry()

    if hasattr(ureg, _from):
        y = _value * ureg(_from)
        return y.to(_to)
    else:
        return "not supported"


def round_float(standard_unit_indices, counts):
    return format(100 * sum(counts[standard_unit_indices]) / sum(counts), ".2f")




def load_json_2_dict(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data


######
def log_weighted_sum_probs(pi_1, log_p1, pi_2, log_p2, pi_3, log_p3):

    x_1 = np.log(pi_1) + log_p1
    x_2 = np.log(pi_2) + log_p2
    x_3 = np.log(pi_3) + log_p3

    xs = [x_1, x_2, x_3]
    log_mx = np.max(xs, axis=0)

    sm = (
        (log_p1 != LOG_EPS) * np.exp(x_1 - log_mx)
        + (log_p2 != LOG_EPS) * np.exp(x_2 - log_mx)
        + (log_p3 != LOG_EPS) * np.exp(x_3 - log_mx)
    )

    return log_mx + np.log(sm)


def log_sum_probs(x):
    log_max_x = np.max(x, axis=0)
    sm = sum([np.exp(x_i - log_max_x) for x_i in x])

    return log_max_x + np.log(sm)


def log_weighted_sum_probs_two(pi_2, log_p2, pi_3, log_p3):

    x_2 = np.log(pi_2) + log_p2
    x_3 = np.log(pi_3) + log_p3

    xs = [x_2, x_3]
    log_mx = np.max(xs, axis=0)

    sm = (log_p2 != LOG_EPS) * np.exp(x_2 - log_mx) + (log_p3 != LOG_EPS) * np.exp(
        x_3 - log_mx
    )

    return log_mx + np.log(sm)


def log_weighted_sum_normalize_probs(pi_1, log_p1, pi_2, log_p2, pi_3, log_p3):

    x_1 = np.log(pi_1) + log_p1
    x_2 = np.log(pi_2) + log_p2
    x_3 = np.log(pi_3) + log_p3

    xs = [x_1, x_2, x_3]
    log_mx = np.max(xs, axis=0)

    sm = (
        (log_p1 != LOG_EPS) * np.exp(x_1 - log_mx)
        + (log_p2 != LOG_EPS) * np.exp(x_2 - log_mx)
        + (log_p3 != LOG_EPS) * np.exp(x_3 - log_mx)
    )
    return x_1, x_2, x_3, log_mx, sm


def normalize_log_probs(probs):
    reduced_probs = np.exp(probs - max(probs))
    return reduced_probs / reduced_probs.sum()


def generate_machine_probabilities(x, symbols, types):
    logP = {}
    for x_i in x:

        temp = [
            np.log(1.0 / len(symbols[t])) if x_i in symbols[t] else LOG_EPS
            for t in types
        ]
        unq_temp = np.unique(temp)
        if len(unq_temp) == 1 and unq_temp[0] == LOG_EPS:
            temp = [
                np.log(1.0 / len(symbols[t])) if x_i.lower() in symbols[t] else LOG_EPS
                for t in types
            ]

        logP[x_i] = (
            [
                0.0 if x_i == "" else LOG_EPS,
            ]
            + [
                np.log(1e-10),
            ]
            + temp
        )
    return logP


def calculate_probabilities(x_unique, unit_ontology, types):
    log_probabilities = {}
    for x_unique_i in x_unique:
        temp_probs = {}
        for t in types:
            temp_probs[t] = {}

            for unit in unit_ontology[t]:
                unit_symbols = unit_ontology[t][unit]
                if (x_unique_i in unit_symbols) or (x_unique_i.lower() in unit_symbols):
                    temp_probs[t][unit] = np.log(1.0 / len(unit_symbols))

        temp_probs["missing"] = 0.0 if x_unique_i == "" else LOG_EPS
        temp_probs["anomaly"] = np.log(1e-30)

        log_probabilities[x_unique_i] = temp_probs

        # for temp in log_probabilities[x_unique_i]:
        #     if log_probabilities[x_unique_i][temp] != {} and temp not in ['missing', 'anomaly']:
        #         print('x_unique_i', x_unique_i, temp, log_probabilities[x_unique_i][temp])
    return log_probabilities


def generate_probs_a_column(x, symbols, types):
    unique_values_in_a_column, counts = np.unique(x, return_counts=True)
    probabilities_dict = generate_machine_probabilities(
        unique_values_in_a_column, symbols, types
    )
    probabilities = np.array(
        [probabilities_dict[x_i] for x_i in unique_values_in_a_column]
    )

    # for value in probabilities_dict:
    #     print(value, probabilities_dict[value], [temp for temp in probabilities_dict[value] if temp != LOG_EPS])
    return probabilities, unique_values_in_a_column, counts


def calculate_likelihoods(x, unit_ontology, types):
    x_unique, x_counts = np.unique(x, return_counts=True)

    probabilities_dict = calculate_probabilities(x_unique, unit_ontology, types)

    return probabilities_dict, x_unique, x_counts


def run_inference(logP, counts, log_prior):
    # Constants
    I, J = logP.shape  # I: num of rows in a data column.
    # J: num of data types including missing and catch-all
    K = J - 2  # K: num of possible column data types (excluding missing and catch-all)

    # Initializations
    pi = [PI for j in range(K)]  # mixture weights of row types

    # Inference
    p_t = []  # p_t: posterior probability distribution of column types
    p_z = np.zeros((I, K, 3))  # p_z: posterior probability distribution of row types

    counts_array = np.reshape(counts, newshape=(len(counts),))

    # Iterates for each possible column type
    for j in range(K):
        # Sum of weighted likelihoods (log-domain)
        p_t.append(
            log_prior[j]
            + (
                counts_array
                * log_weighted_sum_probs(
                    pi[j][0],
                    logP[:, j + LLHOOD_TYPE_START_INDEX],
                    pi[j][1],
                    logP[:, MISSING_INDEX - 1],
                    pi[j][2],
                    logP[:, ANOMALIES_INDEX - 1],
                )
            ).sum()
        )

        # Calculates posterior cell probabilities

        # p_z[:, j, self.TYPE_INDEX] = np.log() +
        # p_z[:, j, self.MISSING_INDEX] = np.log(pi[j][1]) + logP[:,self.MISSING_INDEX-1]
        # p_z[:, j, self.ANOMALIES_INDEX] = np.log(pi[j][2]) + logP[:, self.ANOMALIES_INDEX - 1]

        # Normalizes
        x1, x2, x3, log_mx, sm = log_weighted_sum_normalize_probs(
            pi[j][0],
            logP[:, j + LLHOOD_TYPE_START_INDEX],
            pi[j][1],
            logP[:, MISSING_INDEX - 1],
            pi[j][2],
            logP[:, ANOMALIES_INDEX - 1],
        )

        p_z[:, j, 0] = np.exp(x1 - log_mx - np.log(sm))
        p_z[:, j, 1] = np.exp(x2 - log_mx - np.log(sm))
        p_z[:, j, 2] = np.exp(x3 - log_mx - np.log(sm))
        p_z[:, j, :] = p_z[:, j, :] / p_z[:, j, :].sum(axis=1)[:, np.newaxis]

    p_t = normalize_log_probs(np.reshape(p_t, newshape=(len(p_t),)))
    p_z = p_z

    return [p_t, p_z]


def run_dimension_inference(logP, counts, dimensions):
    K = len(dimensions)

    # Initializations
    pi = [PI for j in range(K)]  # mixture weights of cell dimensions
    log_prior_t = [np.log(1.0 / len(dimensions)) for t in dimensions]
    # log_prior_t[k]: p(t=k)

    # Inference
    p_t = []
    # p_t: p(t|x) posterior probability distribution of column measurement types

    counts_array = np.reshape(counts, newshape=(len(counts),))

    # Iterates for each possible column dimension
    for k, dim in enumerate(dimensions):

        # Sum of weighted likelihoods (log-domain)
        likelihoods = []  # likelihoods = p(t=k|x_{1:I})
        for x_i in logP:
            likelihood_i = 0.0  # likelihood_i = p(t=k|x_i)
            prior_u = 1.0

            if logP[x_i][dim] != {}:  # p( u=l | t=k ) != 0
                for l in logP[x_i][dim]:  # for l in [1, L_t] where p( u=l | t=k ) != 0
                    temp = log_weighted_sum_probs(
                        pi[k][0],
                        logP[x_i][dim][l],
                        pi[k][1],
                        logP[x_i]["missing"],
                        pi[k][2],
                        logP[x_i]["anomaly"],
                    )
                    likelihood_i += prior_u * np.exp(temp)
                    # w_l^l p(x_i|z_i=l) + w_l^m p(x_i|z_i=m) + w_l^a p(x_i|z_i=a)
            else:
                temp = log_weighted_sum_probs(
                    pi[k][0],
                    LOG_EPS,
                    pi[k][1],
                    logP[x_i]["missing"],
                    pi[k][2],
                    logP[x_i]["anomaly"],
                )
                likelihood_i += prior_u * np.exp(temp)

            if likelihood_i == 0.0:
                likelihood_i == LOG_EPS
            likelihoods.append(likelihood_i)
        log_likelihoods = np.log(likelihoods)

        p_t.append(log_prior_t[k] + (counts_array * log_likelihoods).sum())
    p_t = normalize_log_probs(np.reshape(p_t, newshape=(len(p_t),)))

    return p_t


def run_column_unit_inference(u):
    u = [u_i for u_i in u if u_i not in ["missing", "anomaly"]]
    column_unit = Counter(u).most_common(1)[0][0]

    return column_unit


def run_row_units_inference(logP, k, unit_ontology):
    row_units = list(unit_ontology[k].keys())
    u = []
    for x_i in logP:
        p_u_i = run_row_unit_inference(logP, x_i, k, unit_ontology)
        u_i = row_units[np.argmax(p_u_i)]
        u.append(u_i)

    return u


def run_row_unit_inference(logP, x_i, z_i, k, unit_ontology):
    # Initializations

    # log_prior_u[k]: p(u=l|t=k)
    # log_prior_u = [np.log(1.0 / len(unit_ontology[k])) for l in unit_ontology[k]]
    log_prior_u = [np.log(1.0) for l in unit_ontology[k]]

    # Inference
    p_u_i = (
        []
    )  # p_u_i: p(u_i|t, z_i, x_i) posterior probability distribution of row unit

    for l_i, l in enumerate(unit_ontology[k]):
        if (k in logP[x_i]) and (l in logP[x_i][k]) and (z_i == "u_i"):
            log_p = logP[x_i][k][l]
        else:
            log_p = LOG_EPS

        p_u_i.append(log_prior_u[l_i] + log_p)

    p_u_i = normalize_log_probs(np.reshape(p_u_i, newshape=(len(p_u_i),)))

    return p_u_i


def run_row_type_inference(logP, x_i, k, unit_ontology):

    W = PI

    # p_z_i: p(z_i|t, x_i) posterior probability distribution of z_i
    p_z_i = []

    # log_prior_u_i[k]: p(u_i=l|t=k)
    # log_prior_u_i = [np.log(1.0 / len(unit_ontology[k])) for l in unit_ontology[k]]
    log_prior_u_i = [np.log(1.0) for l in unit_ontology[k]]

    if k in logP[x_i]:
        for l in logP[x_i][k]:
            log_p = logP[x_i][k][l]
    else:
        log_p = LOG_EPS

    for i, j in enumerate(["u_i", "missing", "anomaly"]):
        if j == "u_i":
            x = [logP[x_i][k][l] for l in logP[x_i][k]]
        else:
            x = [logP[x_i][j] for l in unit_ontology[k]]

        if x == []:
            p_z_i_j = LOG_EPS
        else:
            p_z_i_j = np.log(W[i]) + log_prior_u_i[0] + log_sum_probs(x)
        p_z_i.append(p_z_i_j)

    p_z_i = normalize_log_probs(np.reshape(p_z_i, newshape=(len(p_z_i),)))

    return p_z_i


def find_same_units(valid_units, units):
    unit_names = {}
    for unit in valid_units:
        names = get_names_for_symbol(unit, units)
        if len(names) != 0:
            unit_name = names[0]
            if unit_name in unit_names:
                unit_names[unit_name].append(unit)
            else:
                unit_names[unit_name] = [
                    unit,
                ]
    matched_units = {}
    for key in unit_names:
        if len(unit_names[key]) > 1:
            print("\tsame unit with different encodings", unit_names[key])
            matched_units[key] = unit_names[key]
    return matched_units


def infer_units(x, symbols, types, units):

    logP, unique_values_in_a_column, counts = generate_probs_a_column(x, symbols, types)

    [p_t, p_z] = run_inference(logP, counts)
    t_predicted = np.argmax(p_t)

    indices = np.where(np.argmax(p_z[:, t_predicted, :], axis=1) == 0)[0]
    print("valid entries =", unique_values_in_a_column[indices])
    same_unit_names = find_same_units(unique_values_in_a_column[indices], units)

    indices = np.where(np.argmax(p_z[:, t_predicted, :], axis=1) == 1)[0]
    print("missing entries =", unique_values_in_a_column[indices])

    indices = np.where(np.argmax(p_z[:, t_predicted, :], axis=1) == 2)[0]
    print("anomalous entries =", unique_values_in_a_column[indices])

    indices = np.where(np.argmax(p_z[:, t_predicted, :], axis=1) == 0)[0]
    alpha = [
        1.0 / len(symbols[types[t_predicted]]) for symbol in symbols[types[t_predicted]]
    ]
    cs = np.zeros((len(symbols[types[t_predicted]]),))

    for i in indices:
        unique_value_in_a_column = unique_values_in_a_column[i]
        cs[symbols[types[t_predicted]].index(unique_value_in_a_column)] = counts[i]
    p_pi = alpha + cs
    pi_predicted = np.argmax(p_pi)
    print("column unit =", symbols[types[t_predicted]][pi_predicted])


# def fast_text_similarity(source, target):
#     return fast_text_model.similarity(source, target)
#
# def scapy_embedding_similarity(source, target):
#     return nlp(source).similarity(nlp(str(target)))


def edit_distance(source, target):
    # INSERTION_COST = 1
    # SUBSTITUTION_COST = 1
    # DELETION_COST = 1
    # LOWERCASE_COST = 0

    INSERTION_COST = 0.5
    SUBSTITUTION_COST = 1.5
    DELETION_COST = 1
    LOWERCASE_COST = 0.25

    if len(source) < len(target):
        return edit_distance(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + INSERTION_COST

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of INSERTION_COST/DELETION_COST),
        # different* (lowercase-uppercase)(cost of LOWERCASE_COST),
        # or are the same (cost of 0).
        temp = []
        for t in target:
            if t != s:
                if t.lower() == s.lower():
                    temp.append(LOWERCASE_COST)
                else:
                    temp.append(SUBSTITUTION_COST)
            else:
                temp.append(0.0)
        current_row[1:] = np.minimum(current_row[1:], np.add(previous_row[:-1], temp))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(current_row[1:], current_row[0:-1] + DELETION_COST)

        previous_row = current_row

    return previous_row[-1]


def edit_distance_header(source, target):
    INSERTION_COST = 1
    DELETION_COST = 1
    LOWERCASE_COST = 0.5
    if len(source) < len(target):
        return edit_distance(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + INSERTION_COST

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of INSERTION_COST/DELETION_COST),
        # different* (lowercase-uppercase)(cost of LOWERCASE_COST),
        # or are the same (cost of 0).
        temp = []
        for t in target:
            if t != s:
                if t.lower() == s.lower():
                    temp.append(LOWERCASE_COST)
                else:
                    temp.append(INSERTION_COST)
            else:
                temp.append(0.0)
        current_row[1:] = np.minimum(current_row[1:], np.add(previous_row[:-1], temp))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(current_row[1:], current_row[0:-1] + DELETION_COST)

        previous_row = current_row

    return previous_row[-1]
