# CONSTANTS FOR PATHS
DATA_ROOT = "experiments/inputs/files new/"
INPUT_ROOT = "experiments/inputs/"
OUTPUT_ROOT = "experiments/outputs/"
DATASETS = [
    "arabica_ratings",
    # "hes",
    "huffman",
    # "phm",
    # "maize_meal",
    # "mba",
    # "open_units",
    # "query_2",
    # "query_4",
    # "robusta_ratings",
    # "zomato",
    # "1438042987662",
    # "1438042986423",
    # "22864497_0_8632623712684511496",
    # "3b5902071ea0f7e0cde6955fb6e474ff",
    # "2015ReportedTaserData",
]


def create_test_path(datafile_name):
    return {
        "data": DATA_ROOT + "test/" + datafile_name,
        "output": OUTPUT_ROOT + "test/" + datafile_name,
    }


DATA_PATHS = {
    "arabica_ratings": {
        "data": DATA_ROOT + "Arabica Ratings.csv",
        # "data": DATA_ROOT + "arabica_ratings_raw.csv",
        "columns": ["Bag Weight", "Altitude"],
        "output": OUTPUT_ROOT + "arabica_ratings.csv",
    },
    "hes": {
        "data": DATA_ROOT + "HES/UKDA-7874-csv/csv/appdata/appliance_data.csv",
        "columns": ["Freezer_volume", "Refrigerator_volume"],
        "output": OUTPUT_ROOT + "hes.csv",
    },
    "huffman": {
        "data": DATA_ROOT + "Huffman.csv",
        "columns": [
            "DISTANCE",
        ],
        "output": OUTPUT_ROOT + "huffman.csv",
    },
    "maize_meal": {
        "data": DATA_ROOT + "Maize Meal.csv",
        "columns": [
            "PACK SIZE1",
        ],
        "output": OUTPUT_ROOT + "maizemeal1.csv",
    },
    "mba": {
        "data": DATA_ROOT + "Market Basket Analysis (MBA).csv",
        "columns": [
            "CURR_SIZE_OF_PRODUCT",
        ],
        "output": OUTPUT_ROOT + "mba.csv",
    },
    "open_units": {
        "data": DATA_ROOT + "Open Units.csv",
        "columns": [
            "Quantity Units",
        ],
        "output": OUTPUT_ROOT + "open_units.csv",
    },
    "phm": {
        "data": DATA_ROOT + "PHM Collection.csv",
        "columns": ["Height", "Weight", "Width", "Depth", "Diameter"],
        "output": OUTPUT_ROOT + "phm.csv",
    },
    "query_2": {
        "data": DATA_ROOT + "WikiData Query_2.csv",
        "columns": ["unitHeightLabel", "unitWidthLabel"],
        "output": OUTPUT_ROOT + "query_2.csv",
    },
    "query_4": {
        "data": DATA_ROOT + "WikiData Query_2.csv",
        "columns": [
            "unitHeightLabel",
        ],
        "output": OUTPUT_ROOT + "query_4.csv",
    },
    "robusta_ratings": {
        "data": DATA_ROOT + "Robusta Ratings.csv",
        "columns": ["Bag Weight", "Altitude"],
        "output": OUTPUT_ROOT + "robusta_ratings.csv",
    },
    "zomato": {
        "data": DATA_ROOT + "Zomato.csv",
        "columns": [
            "currency",
        ],
        "output": OUTPUT_ROOT + "zomato.csv",
    },
    "1438042986423": {
        "data": DATA_ROOT + "T2D 1438042986423.csv",
        "columns": [
            "Size",
        ],
        "output": OUTPUT_ROOT + "1438042986423.csv",
    },
    "1438042987662": {
        "data": DATA_ROOT + "T2D 1438042987662.csv",
        "columns": [
            "FORMAT",
        ],
        "output": OUTPUT_ROOT + "1438042987662.csv",
    },
    "22864497_0_8632623712684511496": {
        "data": DATA_ROOT + "T2D 22864497_0_8632623712684511496.csv",
        "columns": [
            "Size",
        ],
        "output": OUTPUT_ROOT + "22864497_0_8632623712684511496.csv",
    },
    "3b5902071ea0f7e0cde6955fb6e474ff": {
        "data": DATA_ROOT + "GitHub 3b5902071ea0f7e0cde6955fb6e474ff.csv",
        "columns": [
            "amount",
        ],
        "output": OUTPUT_ROOT + "3b5902071ea0f7e0cde6955fb6e474ff.csv",
    },
    "2015ReportedTaserData": {
        "data": DATA_ROOT + "Reported Taser Data.csv",
        "columns": [
            "Height",
        ],
        "output": OUTPUT_ROOT + "2015ReportedTaserData.csv",
    },
}
