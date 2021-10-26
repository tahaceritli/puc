from arpeggio import NoMatch
from collections import Counter
from pint.errors import DefinitionSyntaxError, UndefinedUnitError
from tokenize import TokenError
from quantulum3 import parser

from src.PUC import PUC
from src.utils import parse_cell_value

import numpy as np
import pint
import time

Q_ = pint.UnitRegistry().Quantity
# unitCanonicalizer = UnitCanonicalizer(unit_ontology_path='notebooks/unit-normalization/units_wikidata.json')
unitCanonicalizer = PUC(unit_ontology_path="experiments/inputs/unit_ontology.json")

# from arpeggio import NoMatch

# uncomment the following once S-NER is setup
# from nltk.tag.stanford import StanfordNERTagger
# from nltk.tokenize import word_tokenize
# jar = "/Users/tceritli/Workspace/git/github/aida-repos/processing_meta_data/src/stanford-ner-2018-10-16/stanford-ner.jar"
# model = "/Users/tceritli/Workspace/git/github/aida-repos/processing_meta_data/src/stanford-ner-2018-10-16/data-dictionary.ser.gz"
# trained_ner_tagger = StanfordNERTagger(model, jar, encoding="utf8")

# uncomment the following once CCUT is setup
# import sys
# sys.path.append("ccut/app/ccut_lib/")
# from main.canonical_compound_unit_transformation import CanonicalCompoundUnitTransformation as CCUT
# from qudt.ontology import UnitFactory
# ccut = CCUT()

# uncomment the following once GQ is setup
# from grobid_quantities.quantities import QuantitiesClient
# grobid_server_url = "http://localhost:8060/service"
# client = QuantitiesClient(apiBase=grobid_server_url)

##### cell value predictions - begin #####
def quantulum_predict(_cell_value):
    quants = parser.parse(_cell_value)
    # if len(quants) > 1:
    #     print("more than one quant!", quants)

    if len(quants) == 0:
        return "Not Identified"
    else:
        prediction = {
            "magnitude": quants[0].value,
            "unit": quants[0].unit.name,
            "entity": quants[0].unit.entity.name,
        }
    return prediction


def pint_predict(_cell_value):
    quants = Q_(_cell_value)
    temp = list(quants.dimensionality.keys())
    if len(temp) > 0:
        entity = list(quants.dimensionality.keys())[0].replace("[", "").replace("]", "")
    else:
        entity = "unknown"
    prediction = {
        "magnitude": quants.magnitude,
        "unit": str(quants.units),
        "entity": entity,
    }
    return prediction


def ner_predict(unit):

    # unit = parse_cell_value(_cell_value)[1]
    # print('unit=', unit)
    t = trained_ner_tagger.tag((unit,))[0][1]
    # print('t=', t)
    if unit == "":
        return "dimensionless"
    else:
        return t


def grobid_quantities_predict(_cell_value):
    res = client.process_text(_cell_value)
    if res[0] == 200 and "measurements" in res[1]:

        res = res[1]["measurements"][0]
        if "quantity" in res:
            keyword = "quantity"
        elif "quantityMost" in res:
            keyword = "quantityMost"
        elif "quantityLeast" in res:
            keyword = "quantityLeast"
        elif "quantities" in res:
            keyword = "quantities"

        if keyword == "quantities":
            raw_magnitude = float(res[keyword][0]["rawValue"])
        else:
            raw_magnitude = float(res[keyword]["rawValue"])

        if "rawUnit" in res[keyword]:
            raw_unit = res[keyword]["rawUnit"]["name"]
            raw_unit_name = raw_unit
            if "type" in res[keyword]:
                measurement_type = res[keyword]["type"]
            else:
                measurement_type = "unknown"

            # # get name from symbol
            # raw_unit_name = [rawUnit, measurement_type]
        else:
            raw_unit_name = "Not Identified"
            measurement_type = "unknown"

        prediction = {
            "magnitude": raw_magnitude,
            "unit": raw_unit_name,
            "entity": measurement_type,
        }
    else:
        prediction = "Not Identified"

    return prediction


def puc_predict(y_i, t, u):
    # print("y_i, t, u", y_i, len(y_i), t, u)
    v_i, z_i = unitCanonicalizer.infer_cell_unit(y_i, t, u)
    prediction = {"magnitude": v_i, "unit": z_i}
    return z_i


def ccut_predict(_cell_value):
    canonical_form = ccut.ccu_repr(_cell_value)

    magnitude = canonical_form["ccut:hasPart"][0]["ccut:multiplier"]
    unit = canonical_form["ccut:hasPart"][0]["qudtp:symbol"]

    if unit == "UNKNOWN TYPE":
        prediction = "UNKNOWN TYPE"
    else:
        if "ccut:prefix" in canonical_form["ccut:hasPart"][0]:
            unit = (
                canonical_form["ccut:hasPart"][0]["ccut:prefix"].split("#")[-1]
                + canonical_form["ccut:hasPart"][0]["qudtp:quantityKind"].split("#")[-1]
            )
        else:
            unit = canonical_form["ccut:hasPart"][0]["qudtp:quantityKind"].split("#")[
                -1
            ]

        if unit != "UNKNOWN TYPE":
            qudt_unit = UnitFactory.get_unit("http://qudt.org/vocab/unit#" + unit)
            temp = qudt_unit.type_uri.split("#")
            if len(temp) > 1:
                entity = qudt_unit.type_uri.split("#")[1].replace("Unit", "").lower()
            else:
                entity = "unknown"
        else:
            entity = "unknown"

        prediction = {"magnitude": magnitude, "unit": unit.lower(), "entity": entity}

    return prediction


##### cell value predictions - end #####

##### PUC utils - begin #####
def infer_type_column(df, _column_name):
    x = df[_column_name].to_frame()[_column_name].values
    t, p_z = unitCanonicalizer.infer_column_unit_type(x)
    return t, p_z


def parse_values(y):
    z = [parse_cell_value(y_i) for y_i in y]
    v = [z_i[0] for z_i in z]
    x = [z_i[1] for z_i in z]
    return z, v, x


def generate_likelihoods(x):
    unitCanonicalizer.generate_likelihoods(x)


def infer_column_dimension():
    return unitCanonicalizer.infer_column_dimension()


def infer_cell_units(y, v, x, z, k):
    return unitCanonicalizer.infer_cell_units(y, v, x, z, k)


def infer_cell_types(y, v, x, k):
    return unitCanonicalizer.infer_cell_types(y, v, x, k)


##### PUC utils - end #####

##### run utils - begin #####
def identify_unit_cell(y_i, method, t=None):
    if method == "Pint":
        prediction = pint_predict(y_i)
    elif method == "Quantulum":
        prediction = quantulum_predict(y_i)
    elif method == "CCUT":
        prediction = ccut_predict(y_i)
    elif method == "GQ":
        prediction = grobid_quantities_predict(y_i)
    elif method == "PUC":
        prediction = puc_predict(y_i, t)
    else:
        return "unknown method!"

    return prediction


def identify_cell_unit(x_i, method, t=None, u=None):
    if method == "Pint":
        prediction = pint_predict(x_i)
    elif method == "Quantulum":
        prediction = quantulum_predict(x_i)
    elif method == "CCUT":
        prediction = ccut_predict(x_i)
    elif method == "GQ":
        prediction = grobid_quantities_predict(x_i)
    elif method == "PUC":
        prediction = puc_predict(x_i, t, u)
    else:
        return "unknown method!"

    return prediction


def run_dimension_experiments(df, columns):
    col_dims = {}
    cell_types = {}
    cell_units = {}
    if columns == "all":
        columns = df.columns

    times = {}
    for column in columns:
        y = np.unique(df[column].to_frame()[column].values)
        _, v, x = parse_values(y)

        t0 = time.time()
        generate_likelihoods(x)
        t = infer_column_dimension()
        delta_t = time.time() - t0
        times[column] = delta_t

        if t == "no unit":
            z = "no unit"
            u = "no unit"
        else:
            z = infer_cell_types(y, v, x, t)
            u = infer_cell_units(y, v, x, z, t)

        col_dims[column] = t
        cell_types[column] = z
        cell_units[column] = u

    return col_dims, cell_types, cell_units, times


def run_competitor_column_experiments(df, columns, method):
    if method == "Quantulum":
        return run_quantulum_column_experiments(df, columns)
    elif method == "CCUT":
        return run_ccut_column_experiments(df, columns)
    elif method == "GQ":
        return run_grobid_column_experiments(df, columns)
    elif method == "Pint":
        return run_pint_column_experiments(df, columns)
    elif method == "S-NER":
        return run_ner_column_experiments(df, columns)


def run_quantulum_column_experiments(df, columns):

    predicted_dims = {}
    if columns == "all":
        columns = df.columns
    times = {}
    not_detected = []
    for column in columns:
        temp_dims = []
        unique_values = df[column].unique()
        t0 = time.time()
        for unique_value in unique_values:
            try:
                predicted_dim = quantulum_predict(unique_value)["entity"]
                temp_dims.append(predicted_dim)
            except:
                not_detected.append(unique_value)

        cntr = Counter(temp_dims)
        if len(cntr) == 0:
            t = "unknown"
        else:
            t = cntr.most_common(1)[0][0]
            if t == "dimensionless":
                t = cntr.most_common(2)[1][0]

        delta_t = time.time() - t0
        predicted_dims[column] = t
        times[column] = delta_t

    return predicted_dims, None, times


def run_ccut_column_experiments(df, columns):

    predicted_dims = {}
    if columns == "all":
        columns = df.columns
    times = {}
    undetected_units = []
    for column in columns:
        temp_dims = []
        unique_values = df[column].unique()
        t0 = time.time()
        for unique_value in unique_values:
            try:
                predicted_dim = ccut_predict(unique_value)["entity"]
                temp_dims.append(predicted_dim)
            except:
                undetected_units.append(unique_value)

        cntr = Counter(temp_dims)
        if len(cntr) == 0:
            t = "unknown"
        else:
            t = cntr.most_common(1)[0][0]
            if t == "dimensionless":
                t = cntr.most_common(2)[1][0]

        delta_t = time.time() - t0
        predicted_dims[column] = t
        times[column] = delta_t

    return predicted_dims, None, times


def run_grobid_column_experiments(df, columns):

    predicted_dims = {}
    if columns == "all":
        columns = df.columns
    times = {}
    undetected_units = []

    for column in columns:
        temp_dims = []
        unique_values = df[column].unique()
        t0 = time.time()
        for unique_value in unique_values:
            try:
                predicted_dim = grobid_quantities_predict(unique_value)["entity"]
                temp_dims.append(predicted_dim)
            except:
                undetected_units.append(unique_value)

        cntr = Counter(temp_dims)
        if len(cntr) == 0:
            t = "unknown"
        else:
            t = cntr.most_common(1)[0][0]
            if t == "dimensionless":
                t = cntr.most_common(2)[1][0]

        delta_t = time.time() - t0
        predicted_dims[column] = t
        times[column] = delta_t

    return predicted_dims, None, times


def run_pint_column_experiments(df, columns):
    predicted_dims = {}
    if columns == "all":
        columns = df.columns
    times = {}
    not_detected = []
    for column in columns:
        temp_dims = []
        unique_values = df[column].unique()
        t0 = time.time()
        for unique_value in unique_values:
            try:
                predicted_dim = pint_predict(unique_value)["entity"]
                temp_dims.append(predicted_dim)
            except:
                not_detected.append(unique_value)

        cntr = Counter(temp_dims)
        if len(cntr) == 0:
            t = "unknown"
        else:
            t = cntr.most_common(1)[0][0]
            if t == "dimensionless":
                t = cntr.most_common(2)[1][0]

        delta_t = time.time() - t0
        predicted_dims[column] = t
        times[column] = delta_t

    return predicted_dims, None, times


def run_ner_column_experiments(df, columns):

    predicted_dims = {}
    if columns == "all":
        columns = df.columns

    times = {}
    undetected_units = []
    for column in columns:
        dim_counts = {}
        unique_values = df[column].unique()
        t0 = time.time()
        units = [parse_cell_value(unique_value)[1] for unique_value in unique_values]
        unit_counts = Counter(units)

        for unique_unit in np.unique(units):
            print("processing=", unique_unit)
            try:
                predicted_dim = ner_predict(unique_unit)
                if predicted_dim in dim_counts:
                    dim_counts[predicted_dim] += unit_counts[unique_unit]
                else:
                    dim_counts[predicted_dim] = unit_counts[unique_unit]
            except:
                undetected_units.append(unique_unit)

        if len(dim_counts) == 0:
            t = "unknown"
        else:
            t = max(dim_counts, key=dim_counts.get)
            if t == "dimensionless":
                t = list(sorted(dim_counts.values()))[-2]

        delta_t = time.time() - t0
        predicted_dims[column] = t
        times[column] = delta_t

    return predicted_dims, None, times


def run_identification_experiment(df, cols, method, col_dims=None, cell_types=None):
    predicted_units = {}

    if method == "PUC":
        for col in cols:
            unique_vals = np.unique(df[col].values)
            for cell_val in unique_vals:
                res = identify_cell_unit(
                    cell_val, method, t=col_dims[col], u=cell_types[col]
                )
                predicted_units[cell_val] = res
    else:
        unique_vals = np.unique(df[cols].values)

        for cell_val in unique_vals:
            try:
                res = identify_unit_cell(cell_val, method)
            except UndefinedUnitError:
                res = "UndefinedUnitError"
            except DefinitionSyntaxError:
                res = "DefinitionSyntaxError"
            except ValueError:
                res = "ValueError"
            except FileNotFoundError:
                res = "FileNotFoundError"
            except AttributeError:
                res = "AttributeError"
            except TokenError:
                res = "TokenError"
            except TypeError:
                res = "TypeError"
            except KeyError:
                res = "KeyError"
            except NoMatch:
                res = "NoMatch"

            predicted_units[cell_val] = res

    return predicted_units


##### run utils - end #####
