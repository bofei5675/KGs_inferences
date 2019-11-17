

NUM_ENTITIES = 2191
drugset = set()
with open('./drugbank_data.txt', 'r') as f, open('./drugbank_data_subset{}.txt'.format(NUM_ENTITIES),'a+') as out:

    for line in f:
        drug1, relation, drug2 = line.rstrip().split('\t')

        if drug1 not in drugset and len(drugset) < NUM_ENTITIES:
            drugset.add(drug1)

        if drug2 not in drugset and len(drugset) < NUM_ENTITIES:
            drugset.add(drug2)

        if drug1 in drugset and drug2 in drugset:
            line = line.replace('"', '') # remove a bug in original dataset
            out.write(line)


