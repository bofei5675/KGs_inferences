from lxml import etree
import time
# read data

def main():
    read_start = time.time()
    PREFIX = '{http://www.drugbank.ca}'
    file_name = './data/drug_bank_full_database.xml'
    with open(file_name, 'r') as f:
        tree = etree.parse(f)
    print(f'Parse takes {time.time() - read_start}')


    root = tree.getroot()

    for drug in root:
        drugbank_ids = drug.find(PREFIX + 'drugbank-id')
        primary_id = drugbank_ids.text
        primary_name = drug.find(PREFIX + 'name').text

        drug_interactions = drug.find(PREFIX + 'drug-interactions')
        classification_tag = drug.find(PREFIX + 'classification')
        if classification_tag is None:
            continue
        direct_parent = classification_tag.find(PREFIX + 'direct-parent')
        kingdom = classification_tag.find(PREFIX + 'kingdom')
        super_class = classification_tag.find(PREFIX + 'superclass')
        chem_class = classification_tag.find(PREFIX + 'class')
        sub_class = classification_tag.find(PREFIX + 'subclass')
        with open('drugbank_emb_labels.txt', 'a+') as f:
            f.write(f'{primary_id}\t{direct_parent.text}\t{kingdom.text}\t{super_class.text}\t{chem_class.text}\t{sub_class.text}\n')


if __name__ == '__main__':
    main()
