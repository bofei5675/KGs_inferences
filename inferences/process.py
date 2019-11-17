from lxml import etree
import time
# read data

read_start = time.time()
PREFIX = '{http://www.drugbank.ca}'
file_name = './drug_bank_full_database.xml'
with open(file_name, 'r') as f:
    tree = etree.parse(f)
print(f'Parse takes {time.time() - read_start}')


root = tree.getroot()

for drug in root:
    drugbank_ids = drug.find(PREFIX  + 'drugbank-id')
    primary_id = drugbank_ids.text
    primary_name = drug.find(PREFIX + 'name').text

    drug_interactions = drug.find(PREFIX + 'drug-interactions')

    for drug_interaction in drug_interactions:
        neighbor_name = drug_interaction.find(PREFIX + 'name').text
        neighbor_id = drug_interaction.find(PREFIX + 'drugbank-id').text
        interaction_desc = drug_interaction.find(PREFIX + 'description').text
        interaction_desc = interaction_desc.replace(primary_name,'_')
        interaction_desc = interaction_desc.replace(neighbor_name, '_')
        with open('drugbank_data.txt', 'a+') as f:
            f.write(f'{primary_id}\t{interaction_desc}\t{neighbor_id}\n')




