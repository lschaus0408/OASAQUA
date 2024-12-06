"""
---------------------------------- Observed Antibody Space API -------------------------------------
Module can download different datasets from the OAS and write them into a compressed file for local
storage. Files can be separated into paired and unpaired sequences as well as by their antibody
region. OAS is a database from the University of Oxford Dept. of Statistics (doi: 10.1002/pro.4205)
-------------------------------------- Update Module -----------------------------------------------
Updates the files folder by scraping the available information on OAS and determining all the 
possible queries.
"""

import re
import json
from typing import Iterable, cast
from pathlib import Path
import requests

### ADD UPDATE FOR SUPER_QUERIES


def update_query_check(
    path: Path = Path("./files"),
    query_path: Path = Path("query_check_dictionary.json"),
    cat_path: Path = Path("category_dictionary.json"),
    pattern_1: str = '"form-control custom-select" id="organism" name="',
    pattern_2: str = r">.*<",
    pattern_3: str = "><",
    pattern_4: str = "</select>",
):
    """
    ## Updates Query Check Dictionary
    Requests all possible queries in OAS and packages them into two json files.
    One file mapping all possible categories to queries and the other file
    mapping all queries to a category. This function is necessary for when
    OAS updates names of queries and OASCS throws an exception because the
    query does not exist.
    ### Args: \n
        \tpath{Path} -- path where files are saved \n
        \tquery_path{Path} -- path for query check dict \n
        \tcat_path{Path} -- path for cat dict \n
        \tpattern_1{str} -- where to find the start of the dropdown menu \n
        \tpattern_2{str} -- what the query options are flanked by \n
        \tpattern_3{str} -- what the query options are flanked by without regex \n
        \tpattern_4{str} -- where to find the end of the dropdown menu
    ### Updates:
        \tcategory_dictionary.json -- Mapping of queries to categories \n
        \tquery_check_dictionary.json -- Mapping of categories to queries
    """
    # Update Paired Dict
    response = requests.get(
        "https://opig.stats.ox.ac.uk/webapps/oas/oas_paired/", timeout=10
    )
    paired_dict = extract_queries(response, pattern_1, pattern_2, pattern_3, pattern_4)
    # Update Unpaired Dict
    response = requests.get(
        "https://opig.stats.ox.ac.uk/webapps/oas/oas_unpaired/", timeout=10
    )
    unpaired_dict = extract_queries(
        response, pattern_1, pattern_2, pattern_3, pattern_4
    )

    # Package query dict to json
    query_check_dict = {"unpaired": unpaired_dict, "paired": paired_dict}
    dump_to_json(path.joinpath(query_path), query_check_dict)

    # Package cat dict to json
    category_check_dict = invert_dict_with_list(query_check_dict["unpaired"])
    category_check_dict.update(invert_dict_with_list(query_check_dict["paired"]))
    dump_to_json(path.joinpath(cat_path), category_check_dict)


def invert_dict_with_list(dictionary: dict):
    """
    ## Inverts a dictionary with list values
    """
    return {
        value: key for key, value_list in dictionary.items() for value in value_list
    }


def dump_to_json(path: Path, dictionary: dict):
    """
    ## Dumps provided object to json file set in path
    """
    with open(path, "w", encoding="utf-8") as file:
        json.dump(dictionary, file)


def extract_queries(
    response: requests.Response,
    pattern_1: str,
    pattern_2: str,
    pattern_3: str,
    pattern_4: str,
):
    """
    ## Extracts queries from Request
    """
    query_dict = {}
    for match in re.finditer(pattern=pattern_1, string=response.text):
        # Find keys
        key_start = match.end()
        key_end = key_start + response.text[key_start:].find('"')
        # query_list.append(req.text[key_start:key_end])

        # Find queries
        subqueries = []
        queries_end = key_end + response.text[key_end:].find(pattern_4)
        query_zone = response.text[key_end + 2 : queries_end]
        regex = re.compile(pattern_2)
        subqueries = regex.findall(query_zone)
        subqueries = [item.strip(pattern_3) for item in cast(Iterable[str], subqueries)]
        # Package
        query_dict[response.text[key_start:key_end]] = subqueries
    return query_dict


if __name__ == "__main__":
    pass
