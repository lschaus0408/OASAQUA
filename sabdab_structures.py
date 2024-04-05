import pandas as pd
import numpy as np
from pathlib import Path

SPECIFIC_ANTIGEN = [
    "influenza a virus",
    "influenza b virus",
    "parainfluenza",
    "salmonella enterica",
    "zika virus",
    "human immunodeficiency virus",
    "ebola",
    "west nile virus",
    "homo sapiens",
    "hiv",
    "hepatitis a",
    "hepatitis b",
    "hepatitis c",
    "staphylococcus aureus",
    "severe acute respiratory syndrome",
    "plasmodium falciparum",
    "norovirus",
    "simian immunodeficiency",
    "saccharomyces cerevisiae",
    "human poliovirus",
    "human respiratory syncytial virus",
    "herpesvirus",
    "alphaherpesvirus",
    "human betaherpesvirus",
    "human gammaherpesvirus",
    "human cytomegalovirus",
    "human coronavirus",
    "human betacoronavirus",
    "human adenovirus",
    "human enterovirus",
    "human papillomavirus",
    "human metapneumovirus",
    "human rhinovirus",
    "human parechovirus",
    "human parvovirus",
    "human astrovirus",
    "human sars coronavirus",
    "encephalitis",
    "adeno-associated virus",
    "aequorea victoria",
    "aeropyrum pernix",
    "alkalilimnicola ehrlichii",
    "aquifex aeolicus",
    "arabidopsis",
    "bacillus anthracis",
    "coronavirus",
    "borelia burgdorferi",
    "bos taurus",
    "canine paroviurs",
    "caulobacter vibrioides",
    "centruoides noxius",
    "chaetomium thermophilum",
    "chikungunya virus",
    "clostridium",
    "coxsackievirus",
    "cricetulus",
    "hemorrhagic fever",
    "deinococcus",
    "dengue",
    "dermatophagoides",
    "dickeya",
    "drosophila",
    "echovirus",
    "enterobacteria phage",
    "enterococcus",
    "enterovirus",
    "epstein-barr",
    "escherichia coli",
    "foot-and-mouth disease",
    "haemophilus influenza",
    "hendra",
    "hepacivirus",
    "lactobacillus",
    "lactococcus phage",
    "lactococcus lactis phage",
    "marburgvirus",
    "mammarenavirus",
    "lassa virus",
    "meningitis",
    "macaca",
    "machupo virus",
    "mayaro virus",
    "metarhizium",
    "methanococcus",
    "methanocaldococcus",
    "middle east respiratory",
    "mus musculus",
    "mycobacterium",
    "mycolibacterium",
    "mycoplasma",
    "neisseria gonorrhoeae",
    "neisseria meningitidis",
    "nipah",
    "norwalk virus",
    "oryzias latipes",
    "photobacterium",
    "plasmodium",
    "poliovirus",
    "pseudomonas aeruginosa",
    "rabies",
    "rattus norvegicus",
    "reovirus",
    "respiratory syncytial virus",
    "ricinus communis",
    "rotavirus",
    "saccharopolyspora",
    "salmonella",
    "thrombocytopenia virus",
    "shewanella oneidensis",
    "shigella",
    "staphylococcaceae",
    "staphylococcus",
    "streptococcus",
    "streptomyces",
    "sus scrofa",
    "synthetic construct",
    "thermococcus",
    "thermothelomyces",
    "thermotoga maritima",
    "thermus thermophilus",
    "toxoplasma",
    "trypanosoma",
    "vaccinia",
    "indiana virus",
    "vibrio",
    "xenopus",
    "yellow fever",
    "geobacillus thermodenitrificans",
    "clostridioides",
    "centruroides",
    "campylobacter",
    "borrelia",
    "borreliella",
    "arcobacter",
    "artificial",
    "finegoldia",
    "ross river virus",
]

ANTIGEN_CONSOLIDATION = {
    "influenza a virus": "influenza",
    "influenza b virus": "influenza",
    "parainfluenza": "influenza",
    "human immunodeficiency virus": "HIV",
    "hiv": "HIV",
    "simian immunodeficiency": "HIV",
    "hepatitis a": "hepatitis",
    "hepatitis b": "hepatitis",
    "hepatitis c": "hepatitis",
    "staphylococcaceae": "staphylococcus",
    "staphylococcus aureus": "staphylococcus",
    "staphylococcus": "staphylococcus",
    "severe acute respiratory virus": "coronavirus",
    "human coronavirus": "coronavirus",
    "human betacoronavirus": "coronavirus",
    "sars coronavirus": "coronavirus",
    "coronavirus": "coronavirus",
    "middle east respiratory": "coronavirus",
    "plasmodium falciparum": "plasmodium",
    "plasmodium": "plasmodium",
    "human poliovirus": "poliovirus",
    "poliovirus": "poliovirus",
    "human respiratory syncytial virus": "RSV",
    "respiratory syncytial virus": "RSV",
    "herpesvirus": "herpes",
    "alphaherpesvirus": "herpes",
    "human betaherpesvirus": "herpes",
    "human gammaherpesvirus": "herpes",
    "human adenovirus": "adenovirus",
    "human enterovirus": "enterovirus",
    "enterovirus": "enterovirus",
    "escherichia coli": "ecoli",
    "lactococcus phage": "lactococcus phage",
    "lactococcus lactis phage": "lactococcus phage",
    "neisseria gonorrhoeae": "neisseria",
    "neisseria menigitidis": "neisseria",
}


def main(path: Path, target_path: Path):
    data = pd.read_csv(path, sep="\t")
    data.drop(data[data["antigen_species"] == " | "].index, inplace=True)
    data = data[data["antigen_species"].notna()]
    antigens = data.antigen_species.unique()
    unique_antigens = []
    flag = False
    for antigen in antigens:
        for specific_antigen in SPECIFIC_ANTIGEN:
            if specific_antigen in antigen:
                unique_antigens.append(specific_antigen)
                flag = True
        if not flag:
            unique_antigens.append(antigen)
        flag = False

    unique_antigens = np.unique(unique_antigens)

    for antigen in antigens:
        for unique_antigen in unique_antigens:
            if unique_antigen in antigen:
                data.antigen_species = data.antigen_species.replace(
                    to_replace=antigen, value=unique_antigen
                )
    for antigen in data.antigen_species:
        if antigen in ANTIGEN_CONSOLIDATION:
            data.antigen_species = data.antigen_species.replace(
                to_replace=antigen, value=ANTIGEN_CONSOLIDATION[antigen]
            )
    data.to_csv(target_path, sep="\t")


if __name__ == "__main__":
    path = Path("/home/lschaus/vscode/data/sabdab_summary_all.tsv")
    target_path = Path("/home/lschaus/vscode/data/sabdab_summary_all_processed.tsv")
    main(path, target_path)
