
import time
NUM_ENTITIES = 2191
drugset = set()
tuple_sets = set()
with open('./drugbank_data.txt', 'r') as f, open('./drugbank_data_subset{}.txt'.format(NUM_ENTITIES), 'a+') as out:
    total_lines = 0
    write_lines = 0
    for line in f:
        drug1, relation, drug2 = line.replace('"', '').rstrip().split('\t')

        if drug1 not in drugset and len(drugset) < NUM_ENTITIES:
            drugset.add(drug1)

        if drug2 not in drugset and len(drugset) < NUM_ENTITIES:
            drugset.add(drug2)

        if drug1 in drugset and drug2 in drugset and ' '.join([drug1, relation, drug2]) not in tuple_sets:
            line = line.replace('"', '') # remove a bug in original dataset
            write_lines += 1
            out.write(line)
        else:
            total_lines += 1
            #print('Repeat lines')
            continue

        tuple_sets.add(' '.join([drug1, relation, drug2]))
        tuple_sets.add(' '.join([drug2, relation, drug1]))
        total_lines += 1
        #print(tuple_sets)
        #time.sleep(3)
print('Total:', total_lines, 'Write', write_lines)


