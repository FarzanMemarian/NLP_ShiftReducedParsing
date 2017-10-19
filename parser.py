# parser.py

import sys
from models import *
from random import shuffle
from pdb import set_trace
import argparse

if __name__ == '__main__':

    a = argparse.ArgumentParser()
    a.add_argument("--system_to_run", default="GREEDY")
    a.add_argument("--nb_exm", default=1000, type=int)
    a.add_argument("--beam_size", default=5, type=int)
    a.add_argument("--early_stopping", default=True)
    a.add_argument("--extra_features", default=False)
    a.add_argument("--run_on_test", default=True)
    a.add_argument("--epochs", default=5 ,type=int)

    args = a.parse_args()

    # Load the training and test data
    print "Reading train data..."
    train_whole = read_data("data/train.conllx")
    train = train_whole[:args.nb_exm]
    shuffle(train)
    print "Kept " + repr(len(train)) + " exs"
    print "Reading dev data..."
    dev_whole = read_data("data/dev.conllx")
    dev = dev_whole[:]
    # Here's a few sentences...
    print "Examples of sentences:"
    # print str(dev[1])
    # print str(dev[3])
    # print str(dev[5])
    
    # Set to true to produce final output



    if args.system_to_run == "TEST_TRANSITIONS":
        for idx in xrange(0, len(dev)):
            parsed_sentence = dev[idx]
            print "INDEX: " + repr(idx)
            (decisions, states) = get_decision_sequence(parsed_sentence)
            parsed_dev.append(ParsedSentence(parsed_sentence.tokens, states[-1].get_dep_objs(len(parsed_sentence))))
    elif args.system_to_run == "GREEDY":
        trained_model = train_greedy_model(train, args.extra_features, args.epochs)
        print "Parsing dev"
        parsed_dev = [trained_model.parse(sent) for sent in dev]
        if args.run_on_test:
            print "Parsing test"
            test = read_data("data/test.conllx.blind")
            test_decoded = [trained_model.parse(test_ex) for test_ex in test]
            print_output(test_decoded, "test.conllx.out")
    elif args.system_to_run == "BEAM":
        trained_model = train_beamed_model(train, args.early_stopping, args.beam_size, args.extra_features, args.epochs)
        print "Parsing dev"
        parsed_dev = [trained_model.parse(sent)[0] for sent in dev]
        if args.run_on_test:
            print "Parsing test"
            test = read_data("data/test.conllx.blind")
            test_decoded = [trained_model.parse(test_ex) for test_ex in test]
            print_output(test_decoded, "test.conllx.out")
    else:
        raise Exception("Pass in either TEST_TRANSITIONS, GREEDY, or BEAM to run the appropriate system")
    print_evaluation(dev, parsed_dev)
