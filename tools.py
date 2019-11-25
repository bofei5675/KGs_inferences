def save_mbeddings(args, entity_emb, relation_emb):

    with open(args.output_folder + 'entity_emb.txt', 'w') as f:
        for i in range(len(entity_emb)):
            vec = entity_emb[i]
            f.write("{} {}\n".format(args.id2entity[i], ' '.join([str(x) for x in vec])))

    with open(args.output_folder + 'relation_emb.txt', 'w') as f:
        for i in range(len(relation_emb)):
            vec = relation_emb[i]
            f.write("{} {}\n".format(args.id2relation[i], ' '.join([str(x) for x in vec])))

