import json
import numpy as np

from src.utils import (
    get_all_entities,
    get_names_for_symbol,
    get_symbols_for_entity,
    get_num,
    get_unit,
    generate_probs_a_column,
    calculate_likelihoods,
    run_inference,
    run_dimension_inference,
    run_column_unit_inference,
    run_row_unit_inference,
    run_row_type_inference,
    round_float,
    find_same_units,
    convert_value,
    load_json_2_dict,
    edit_distance,
    edit_distance_header,
    parse_cell_value,
    log_sum_exp,
)

# from qudt.ontology import UnitFactory
# from wikidata.client import Client


class PUC:
    def __init__(self, unit_ontology_path="experiments/inputs/unit_ontology.json"):

        self.unit_ontology = load_json_2_dict(unit_ontology_path)
        self.dimensions = list(self.unit_ontology.keys())

    def generate_likelihoods(self, x):
        log_probabilities, x_unique, x_counts = calculate_likelihoods(
            x, self.unit_ontology, self.dimensions
        )

        self.log_probabilities = log_probabilities
        self.x_unique = x_unique
        self.x_counts = x_counts

    def convert_row_unit(self, v_i, u_i, u):
        return convert_value(v_i, u_i, u)

    def infer_column_dimension(
        self,
    ):
        p_t = run_dimension_inference(
            self.log_probabilities, self.x_counts, self.dimensions
        )

        if len(np.unique(p_t)) == 1:
            return "no unit"
        else:
            return self.dimensions[np.argmax(p_t)]

    def infer_column_unit(self, u):
        column_unit = run_column_unit_inference(u)
        return column_unit

    def infer_cell_unit(self, x_i, z_i, t):
        row_units = list(self.unit_ontology[t].keys())
        p_u_i = run_row_unit_inference(
            self.log_probabilities, x_i, z_i, t, self.unit_ontology
        )
        # print('p_u_i', p_u_i)

        # if it's uniform, then it's either missing or anomalous
        if len(np.unique(p_u_i)) == 1:
            u_i = z_i
        else:
            u_i = row_units[np.argmax(p_u_i)]
        return u_i

    def infer_cell_units(self, y, v, x, z, t):
        predictions = {}
        for y_i, v_i, x_i in zip(y, v, x):
            z_i = z[y_i]
            u_i = self.infer_cell_unit(x_i, z_i, t)

            if u_i == "anomaly":
                u_i = self.map_anomalous_symbols(x_i, t)

            predictions[y_i] = {"magnitude": v_i, "unit": u_i}

        return predictions

    def infer_cell_dimension(self, x_i, t):
        row_units = ["u_i", "missing", "anomaly"]
        p_z_i = run_row_type_inference(
            self.log_probabilities, x_i, t, self.unit_ontology
        )
        z_i = row_units[np.argmax(p_z_i)]
        return z_i

    def infer_cell_dimensions(self, y, v, x, t):
        predictions = {}
        for y_i, v_i, x_i in zip(y, v, x):
            z_i = self.infer_cell_dimension(x_i, t)
            predictions[y_i] = z_i

        return predictions

    def map_anomalous_symbols(self, x_i, k):
        min_distance = 10000
        unit_mapped = "anomaly"
        for unit in self.unit_ontology[k]:
            unit_symbols = self.unit_ontology[k][unit]

            for unit_symbol in unit_symbols:
                dist = edit_distance(x_i, unit_symbol)

                if dist < min_distance:
                    min_distance = dist
                    unit_mapped = unit

        return unit_mapped

    def canonicalize_explicit_units(self, y):
        pass

    def search_wiki(self, _cell_value, _cell_type=None):
        matches = []

        # find the matched literal
        for entity_id in self.wiki_units:
            wiki_unit = self.wiki_units[entity_id]
            m_type = wiki_unit[0]
            u_name = wiki_unit[1]

            # this is a wikidata mistake it seems
            if u_name == "To":
                continue

            u_literals = wiki_unit[2]

            if (_cell_value in u_literals and _cell_type == None) or (
                _cell_value in u_literals and _cell_type == m_type
            ):
                temp = u_literals
                matches.append(temp)

        if len(matches) == 0:  # this is not a very good idea
            matches = get_names_for_symbol(_cell_value, self.units)
            if len(matches) == 0:
                temp = "no unit"
            else:
                temp = matches[0]
        elif len(matches) == 1:
            temp = matches[0][0]
        else:
            index = 999
            for match in matches:
                proposed_index = match.index(_cell_value)
                if proposed_index < index:
                    index = proposed_index
                    tempp = match
            temp = tempp[0]

        return temp

    def search_quantulum_units(self, _symbol, _cell_type):
        return get_names_for_symbol(_symbol, _cell_type, self.units)

    def identify_unit_cell(self, _cell_value, _cell_type):

        y, x = parse_cell_value(_cell_value)
        # y = get_num(_cell_value)
        # x = get_unit(_cell_value)
        # print('y', y, 'x', x, '_cell_type', _cell_type, '_cell_value', _cell_value)
        if _cell_type == "no unit":
            # how to search when the measurement type is unknown.
            # temp = self.search_wiki(x)
            temp = "no unit"
            # if x in self.symbols[_cell_type]:
            #
            # elif x.lower() in self.symbols[_cell_type]:
            #     temp = self.search_wiki(x.lower())
        else:
            if x == "" or x.isspace():
                temp = "no unit"
            elif x in self.symbols[_cell_type]:
                print("x", x, "self.symbols[_cell_type]", self.symbols[_cell_type])
                temp = self.search_quantulum_units(x, _cell_type)
                if len(temp) == 0:
                    temp = self.search_wiki(x, _cell_type)
                print("temp", temp)
            elif x.lower() in self.symbols[_cell_type]:
                temp = self.search_wiki(x.lower(), _cell_type)
            else:
                # check string-similarities given _cell_type
                print("through string-similarities  model")
                distances = {}
                for entity_id in self.wiki_units:
                    wiki_unit = self.wiki_units[entity_id]
                    m_type = wiki_unit[0]
                    u_name = wiki_unit[1]
                    u_literals = wiki_unit[2]

                    # this is a wikidata mistake it seems
                    if u_name == "To":
                        continue

                    if m_type == _cell_type:
                        literals_distance = []
                        for u_literal in u_literals:
                            literals_distance.append(edit_distance(x, u_literal))
                        min_distance = np.min(
                            literals_distance
                        )  # /len(assigned_distances)
                        count_min_distance = literals_distance.count(min_distance)
                        distances[u_name] = [min_distance, count_min_distance]

                        print(u_name, u_literals, literals_distance)

                if len(distances) != 0:
                    min_dist = 999
                    count = 0
                    for u_name in distances:
                        if (distances[u_name][0] < min_dist) or (
                            distances[u_name][0] == min_dist
                            and count < distances[u_name][1]
                        ):
                            print(u_name, distances[u_name])
                            min_dist = distances[u_name][0]
                            count = distances[u_name][1]
                            temp = u_name
                    print("a new unit", x, "is matched to", temp)
                else:
                    temp = "no unit"
                    # for _cell_type in self.symbols:
                    #     if x in self.symbols[_cell_type]:

        return [y, temp]

    def identify_unit_cell_exponential(self, _cell_value, _cell_type):

        y, x = parse_cell_value(_cell_value)
        # y = get_num(_cell_value)
        # x = get_unit(_cell_value)
        # print('y', y, 'x', x, '_cell_type', _cell_type, '_cell_value', _cell_value)
        if _cell_type == "no unit":
            # how to search when the measurement type is unknown.
            # temp = self.search_wiki(x)
            temp = "no unit"
            # if x in self.symbols[_cell_type]:
            #
            # elif x.lower() in self.symbols[_cell_type]:
            #     temp = self.search_wiki(x.lower())
        else:
            if x == "" or x.isspace():
                temp = "no unit"
            elif x in self.symbols[_cell_type]:
                print("x", x, "self.symbols[_cell_type]", self.symbols[_cell_type])
                temp = self.search_quantulum_units(x, _cell_type)
                if len(temp) == 0:
                    temp = self.search_wiki(x, _cell_type)
                print("temp", temp)
            elif x.lower() in self.symbols[_cell_type]:
                temp = self.search_wiki(x.lower(), _cell_type)
            else:
                # check string-similarities given _cell_type
                print("through string-similarities  model")
                gamma = 2
                u_t = self.symbols[_cell_type]
                likelihoods = [-gamma * edit_distance(x, u_tl) for u_tl in u_t]
                # likelihoods = log_sum_exp(x)
                print("likelihoods", likelihoods)
                print("np.argmax(likelihoods)", np.argmax(likelihoods))
                most_likely_known_unit_symbol = u_t[np.argmax(likelihoods)]
                print("most_likely_known_unit_symbol", most_likely_known_unit_symbol)

                # what is the name of this unit?
                most_likely_names = self.search_quantulum_units(
                    most_likely_known_unit_symbol, _cell_type
                )

                for entity_id in self.wiki_units:
                    wiki_unit = self.wiki_units[entity_id]
                    m_type = wiki_unit[0]
                    u_name = wiki_unit[1]
                    u_literals = wiki_unit[2]

                    # this is a wikidata mistake it seems
                    if u_name == "To":
                        continue

                    if (m_type == _cell_type) and (
                        most_likely_known_unit_symbol in u_literals
                    ):
                        most_likely_names.append(u_name)
                print("a new unit", x, "is matched to", most_likely_names)

                if len(most_likely_names) == 0:
                    temp = "no unit"
                elif len(most_likely_names) == 1:
                    temp = most_likely_names[0]
                else:
                    temp = most_likely_names[0]
                    print("dikkat! multiple unit names matched")

                    # for _cell_type in self.symbols:
                    #     if x in self.symbols[_cell_type]:

        return [y, temp]

    def identify_row_unit(self, _cell_value, _cell_type):

        y, x = parse_cell_value(_cell_value)

        # y = get_num(_cell_value)
        # x = get_unit(_cell_value)
        # print('y', y, 'x', x, '_cell_type', _cell_type, '_cell_value', _cell_value)
        if _cell_type == "no unit":
            # how to search when the measurement type is unknown.
            # temp = self.search_wiki(x)
            temp = "no unit"
            # if x in self.symbols[_cell_type]:
            #
            # elif x.lower() in self.symbols[_cell_type]:
            #     temp = self.search_wiki(x.lower())
        else:
            if x == "" or x.isspace():
                temp = "no unit"
            elif x in self.symbols[_cell_type]:
                print("x", x, "self.symbols[_cell_type]", self.symbols[_cell_type])
                temp = self.search_quantulum_units(x, _cell_type)
                if len(temp) == 0:
                    temp = self.search_wiki(x, _cell_type)
                print("temp", temp)
            elif x.lower() in self.symbols[_cell_type]:
                temp = self.search_wiki(x.lower(), _cell_type)
            else:
                # check string-similarities given _cell_type
                print("through string-similarities  model")
                gamma = 2
                u_t = self.symbols[_cell_type]
                likelihoods = [-gamma * edit_distance(x, u_tl) for u_tl in u_t]
                # likelihoods = log_sum_exp(x)
                print("likelihoods", likelihoods)
                print("np.argmax(likelihoods)", np.argmax(likelihoods))
                most_likely_known_unit_symbol = u_t[np.argmax(likelihoods)]
                print("most_likely_known_unit_symbol", most_likely_known_unit_symbol)

                # what is the name of this unit?
                most_likely_names = self.search_quantulum_units(
                    most_likely_known_unit_symbol, _cell_type
                )

                for entity_id in self.wiki_units:
                    wiki_unit = self.wiki_units[entity_id]
                    m_type = wiki_unit[0]
                    u_name = wiki_unit[1]
                    u_literals = wiki_unit[2]

                    # this is a wikidata mistake it seems
                    if u_name == "To":
                        continue

                    if (m_type == _cell_type) and (
                        most_likely_known_unit_symbol in u_literals
                    ):
                        most_likely_names.append(u_name)
                print("a new unit", x, "is matched to", most_likely_names)

                if len(most_likely_names) == 0:
                    temp = "no unit"
                elif len(most_likely_names) == 1:
                    temp = most_likely_names[0]
                else:
                    temp = most_likely_names[0]
                    print("dikkat! multiple unit names matched")

                    # for _cell_type in self.symbols:
                    #     if x in self.symbols[_cell_type]:

        return [y, temp]

    def infer_units(self, df, column_name):
        x = df[column_name].to_frame()[column_name].values
        y = [get_num(x_i) for x_i in x]
        x = [get_unit(x_i) for x_i in x]

        logP, unique_values_in_a_column, counts = generate_probs_a_column(
            x, self.symbols, self.dimensions
        )

        [p_t, p_z] = run_inference(logP, counts)
        if len(np.unique(p_t)) == 1:
            print("flat posterior is calculated. no units in the column?")
            return df
        else:
            t_predicted = np.argmax(p_t)
            print("unit type of the column is", self.dimensions[t_predicted], "\n")

            standard_unit_indices = np.where(
                np.argmax(p_z[:, t_predicted, :], axis=1) == 0
            )[0]
            standard_unit_percentage = round_float(standard_unit_indices, counts)

            missing_unit_indices = np.where(
                np.argmax(p_z[:, t_predicted, :], axis=1) == 1
            )[0]
            missing_unit_percentage = round_float(missing_unit_indices, counts)

            non_standard_unit_indices = np.where(
                np.argmax(p_z[:, t_predicted, :], axis=1) == 2
            )[0]
            non_standard_unit_percentage = round_float(
                non_standard_unit_indices, counts
            )

            print(
                "standard units found =",
                unique_values_in_a_column[standard_unit_indices],
                standard_unit_percentage,
            )
            print(
                "non-standard units found =",
                unique_values_in_a_column[non_standard_unit_indices],
                non_standard_unit_percentage,
            )
            print(
                "missing entries =",
                unique_values_in_a_column[missing_unit_indices],
                missing_unit_percentage,
            )

            # checking the representations of the same unit
            same_unit_names = find_same_units(
                unique_values_in_a_column[standard_unit_indices], self.units
            )

            indices = np.where(np.argmax(p_z[:, t_predicted, :], axis=1) == 0)[0]
            alpha = [
                1.0 / len(self.symbols[self.dimensions[t_predicted]])
                for symbol in self.symbols[self.dimensions[t_predicted]]
            ]
            cs = np.zeros((len(self.symbols[self.dimensions[t_predicted]]),))

            for i in indices:
                unique_value_in_a_column = unique_values_in_a_column[i]
                cs[
                    self.symbols[self.dimensions[t_predicted]].index(
                        unique_value_in_a_column
                    )
                ] = counts[i]
            p_pi = alpha + cs
            pi_predicted = np.argmax(p_pi)
            most_common_unit = self.symbols[self.dimensions[t_predicted]][pi_predicted]
            print("\nmost common unit is", most_common_unit)

            # check whether there is any standard unit, not supported by pint
            unit_maps = {}
            for unique_unit in unique_values_in_a_column[standard_unit_indices]:
                temp = convert_value(1.0, unique_unit, most_common_unit)
                if temp == "not supported":
                    print("pint does not support", unique_unit)

                    # check whether there is another unit, e.g. g for gm
                    for key in same_unit_names:
                        if unique_unit in same_unit_names[key]:
                            alternative_units = list(
                                set(same_unit_names[key])
                                - set(
                                    [
                                        unique_unit,
                                    ]
                                )
                            )

                            for alternative_unit in alternative_units:
                                temp = convert_value(
                                    1.0, alternative_unit, most_common_unit
                                )
                                if temp != "not supported":
                                    unit_maps[unique_unit] = alternative_unit
                                    print(
                                        unique_unit,
                                        "can be matched to",
                                        alternative_unit,
                                    )
                                    break
            x_converted = []
            y_converted = []
            num_conversions = 0
            for x_i, y_i in zip(x, y):
                if (
                    x_i in unique_values_in_a_column[standard_unit_indices]
                    and x_i != most_common_unit
                ):

                    if x_i in unit_maps:
                        y_i_converted = convert_value(
                            float(y_i), unit_maps[x_i], most_common_unit
                        ).magnitude
                    else:
                        y_i_converted = convert_value(
                            float(y_i), x_i, most_common_unit
                        ).magnitude

                    num_conversions += 1
                    x_converted.append(most_common_unit)
                    y_converted.append(y_i_converted)
                else:
                    x_converted.append(x_i)
                    y_converted.append(y_i)

            print(
                "number of conversions from standard units to "
                + most_common_unit
                + " =",
                num_conversions,
                "\n",
            )

            df_new = df.copy()
            df_new[column_name + " (" + most_common_unit + ")"] = [
                str(y_i) + " " + x_i for x_i, y_i in zip(x_converted, y_converted)
            ]

            return df_new

    def infer_column_unit_type(self, x, column, feature):
        z = [parse_cell_value(x_i) for x_i in x]
        y = [z_i[0] for z_i in z]
        x = [z_i[1] for z_i in z]

        print("x=", x)

        # y = [get_num(x_i) for x_i in x]
        # x = [get_unit(x_i) for x_i in x]

        # calculate log prior probabilities using column and self.types
        # coeffs = []
        # for t in self.types:
        #     min_dist = 999
        #     for token in column.split(' '):
        #         current_dist = edit_distance_header(token, t)

        #         if current_dist < min_dist:
        #             min_dist = current_dist
        #     coeffs.append(min_dist)
        print("feature", feature)
        if feature == "no header":
            log_prior = [np.log(1.0 / len(self.dimensions)) for t in self.dimensions]
        elif feature == "string_similarity":
            coeffs = [edit_distance(column, t) for t in self.dimensions]
            log_prior = [np.log(1.0 - coeff / sum(coeffs)) for coeff in coeffs]
        elif feature == "embedding_similarity":
            # coeffs = [scapy_embedding_similarity(column, t) for t in self.types]
            coeffs = [fast_text_similarity(column, t) for t in self.dimensions]
            if sum(coeffs) == 0:
                log_prior = [np.log(1.0 / len(coeffs)) for coeff in coeffs]
            else:
                log_prior = [np.log(coeff / sum(coeffs)) for coeff in coeffs]
        elif feature == "both":
            coeffs = [edit_distance(column, t) for t in self.dimensions]
            prior_string = [1.0 - coeff / sum(coeffs) for coeff in coeffs]

            # coeffs = [scapy_embedding_similarity(column, t) for t in self.types]
            coeffs = [fast_text_similarity(column, t) for t in self.dimensions]
            if sum(coeffs) == 0:
                prior_embedding = [1.0 / len(coeffs) for coeff in coeffs]
            else:
                prior_embedding = [coeff / sum(coeffs) for coeff in coeffs]

            log_prior = [
                np.log((prior_s + prior_e) / 2.0)
                for prior_s, prior_e in zip(prior_string, prior_embedding)
            ]
        else:
            print("Undefined feature!")
        # print('only header with gensim', column, self.types[np.argmax(log_prior)])

        # log_prior = [np.log(edit_distance(column, t)) for t in self.types]
        # print('only header with edit distance', column, self.types[np.argmax(log_prior)])
        logP, unique_values_in_a_column, counts = generate_probs_a_column(
            x, self.symbols, self.dimensions
        )
        [p_t, p_z] = run_inference(logP, counts, log_prior)
        print("p_t", p_t)
        if len(np.unique(p_t)) == 1:
            return "no unit", "no unit"
        else:
            return self.dimensions[np.argmax(p_t)], p_z

    def infer_column_unit_symbol(self, z, t_predicted, cell_prediction):
        print(z)
        x = []
        for x_i in z:
            if (x_i != "") and (not x_i.isspace()):
                u_symbol = None
                temp = cell_prediction[x_i]["unit"]

                if type(temp) == list:
                    temp = temp[0]

                for quantulum_unit in self.units:
                    # print('x_i=', x_i)
                    # print('temp=', temp)
                    # print('quantulum_unit=', quantulum_unit)
                    if (
                        (temp in quantulum_unit["name"])
                        or (
                            "surfaces" in quantulum_unit
                            and temp in quantulum_unit["surfaces"]
                        )
                        or (
                            "symbols" in quantulum_unit
                            and temp in quantulum_unit["symbols"]
                        )
                    ):
                        if len(quantulum_unit["symbols"]) != 0:
                            u_symbol = quantulum_unit["symbols"][0]
                            # print('u_symbol', u_symbol)
                            break
                if u_symbol is None:
                    x.append(parse_cell_value(x_i)[1])
                else:
                    x.append(u_symbol)

        print(x)
        # x = [parse_cell_value(x_i)[1] for x_i in z]

        unique_values_in_a_column, counts = np.unique(x, return_counts=True)

        alpha = [
            1.0 / len(self.symbols[t_predicted]) for symbol in self.symbols[t_predicted]
        ]
        cs = np.zeros((len(self.symbols[t_predicted]),))
        for unique_value_in_a_column, count in zip(unique_values_in_a_column, counts):

            if unique_value_in_a_column in self.symbols[t_predicted]:
                cs[self.symbols[t_predicted].index(unique_value_in_a_column)] += count
            else:
                print(
                    "cell_unit_symbol_predicted",
                    unique_value_in_a_column,
                    "not in self.symbols[t_predicted]:",
                )

        p_pi = alpha + cs
        pi_predicted = np.argmax(p_pi)
        most_common_unit = self.symbols[t_predicted][pi_predicted]
        print("\nmost common unit is", most_common_unit)

        return most_common_unit

    def infer_unit_types(self, df):
        predicted_types = []
        for column_name in df.columns:
            x = df[column_name].to_frame()[column_name].values
            y = []
            for x_i in x:
                y.append(get_unit(str(x_i)))
                # try:
                #
                # except:
                #     continue
            x = y
            # x = [get_unit(x_i) for x_i in x]
            if x == []:
                predicted_types.append("no units")
            else:
                logP, unique_values_in_a_column, counts = generate_probs_a_column(
                    x, self.symbols, self.dimensions
                )

                [p_t, p_z] = run_inference(logP, counts)

                if len(np.unique(p_t)) == 1:
                    predicted_types.append("no units")
                    # predicted_types[column_name] = 'no units'
                else:
                    t_predicted = np.argmax(p_t)
                    predicted_types.append(self.dimensions[t_predicted])
                    # predicted_types[column_name] = self.types[t_predicted]

        return predicted_types
