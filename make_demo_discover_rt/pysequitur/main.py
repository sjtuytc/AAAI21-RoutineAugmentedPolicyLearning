#!/usr/local/bin/python
# -*- coding: utf-8 -*-
# file: main.py

from .sequiturpython.grammar import Grammar, Symbol
from .sequiturpython.symbol import RuleIndex, RULE_INDEX_STR

# Few constants for presentation logics
#RULE_INDEX_STR = "^%s"
SEQUENCE_KEY = "S"
ARROW = "→"
NEWLINE_REPLACEMENT = "↵"
SPACE_REPLACEMENT = "_"
TAB_REPLACEMENT = "↹"


class AlphabetsTransformer:
    def __init__(self):
        self.alphabets_encoder = [chr(num) for num in range(1000)]
        self.alphabets_decoder = {key: idx for idx, key in enumerate(self.alphabets_encoder)}

    def list_ids2alphabets(self, one_list):
        return [self.alphabets_encoder[cur_ele] for cur_ele in one_list]

    def list_alphabets2ids(self, one_list):
        return [self.alphabets_decoder[cur_ele] for cur_ele in one_list]


class Rule(list):
    """ Rule class keeps track of digrams on a list """
    def __new__(cls, v=[]):
        obj = list.__new__(cls, [v])
        obj.c = 0 # set default counter value
        obj.i = RuleIndex(0) # set default index value
        return obj
    
    def ind(self, i=None):
        """ Set and get index """
        if i is not None:
            self.i = RuleIndex(i)
        return self.i
    
    def inc(self, n=1):
        """ Increase counter """
        self.c += n
        return self.c
    
    def dec(self, n=1):
        """ Decrease counter """
        self.c -= n
        return self.c
    
    def cnt(self):
        """ Get counter """
        return self.c
    
    def replace(self, rule):
        """
        Replace rule digram values by other rule digrams. This is not used for Sequencer2!
        If self rule is: [[1, 2], [2, 3]] and 1 is replaced with rule given on argument: [['a', 'b'], ['b', 'c']]
        would become: [['a', 'b'], ['b', 'c'], ['c', 2], [2, 3]]
        """
        for ind, digram in enumerate(self):
            # Digram has two values, potentially rule indexes
            # both of them must be compared with the given rule index
            for j, el in enumerate(digram):
                ind += j # j = 0 or 1
                if isinstance(el, RuleIndex) and el == rule.ind():
                    if ind > 0:
                        self[ind-1][1] = rule[0][0]
                    if ind < len(self):
                        self[ind][0] = rule[-1][1]
                    self[ind:ind] = rule[:]


class Sequencer(list):
    """ Main class to use algorithm. This implements the digram based approach for the algo. """
    def __init__(self, seq=[], utilize=True):
        self.first = None
        if seq:
            for c in seq:
                self.stream(c, utilize)
    
    def utilize(self):
        """ Remove redundant rules i.e. if rule is used only once on rules """
        rules = self[1:]
        for rule1 in rules:
            # only rules with count = 1
            if rule1 is None or rule1.cnt() != 1:
                continue
            for rule2 in rules:
                # iterate over all rules except the excluded rule and None
                if rule2 is None or rule2 is rule1:
                    continue
                rule2.replace(rule1)
            # free up the slot for the next reoccurring rule
            self[rule1.ind()] = None
    
    def find(self, digram):
        """ Find given digram from main rule / sequence and rest of the rules """
        for i, rule in enumerate(self):
            if rule is None:
                continue
            # main rule
            if i == 0:
                j = rule.index(digram) if digram in rule[:-1] else None
                if j is not None:
                    return 0, j, -1
            # rules with one digram
            elif len(rule) == 1:
                if rule[0] == digram:
                    return i, 0, -1
            # rules with multiple digrams
            else:
                j = rule.index(digram) if digram in rule else None
                if j is not None:
                    return i, j, 1
        return (-1, -1, -1)
    
    def new_rule(self, rule):
        """ New rule creator helper """
        # get new index from empty slots if available
        if None in self:
            c = rule.ind(self.index(None))
            self[c] = rule
        # else get new index from total length of the sequence
        else:
            c = rule.ind(len(self))
            self.append(rule)
        return c
    
    def stream(self, c, utilize=True):
        """ Main sequence handler / algorithm """
        # create first item, if not exists yet
        if self.first is None:
            self.first = c
            r = [[None, c]]
            self.append(Rule(r))
            return
        
        main = self[0]
        util = False
        # loop as many times as there are no more repeating digrams
        while True:
            # create a new digram from previous digram last item and coming item c
            digram = [main[-1][1], c]
            # search if main sequence of rest of the rules has the digram
            ind, j, k = self.find(digram)
            # rule is a list of digrams, the first digram is instantiated here
            rule = Rule([digram])
            # digram found from main rule
            if ind == 0:
                # increase potential previous rule index
                if isinstance(c, RuleIndex):
                    self[c].inc()
                # get a new item by rule creation
                c = self.new_rule(rule)
                # every new rule will get counter increased by two
                self[c].inc(2)
                # decrease counter of the replaced rules
                if isinstance(main[j-1][1], RuleIndex):
                    self[main[j-1][1]].dec()
                    util = True
                if isinstance(main[j+1][0], RuleIndex):
                    self[main[j+1][0]].dec()
                    util = True
                # replace certain items with a new rule item: c
                main[-1][1] = main[j+1][0] = main[j-1][1] = c
                del main[j]
                # break while loop
                break
            else:
                # digram was not found from the main sequence, but is found from the other rules
                if ind > 0:
                    # digram was found especially from longer rules, i.e. rules that are longer than one digram long
                    if k > 0:
                        # get a new item by rule creation
                        c = self.new_rule(rule)
                        # increase counter
                        rule.inc()
                        # change rule content by adding new index
                        if j < len(self[ind])-1:
                            self[ind][j+1][0] = c
                        if j-1 > -1:
                            self[ind][j-1][1] = c
                        # delete old rule digram
                        del self[ind][j]
                    else:
                        # create index for the next digram
                        c = RuleIndex(ind)
                    # remove last item from the main sequence
                    l = main.pop()
                    # if the rightmost value of the removed rule is a RuleIndex, decrease counter
                    if isinstance(l[1], RuleIndex):
                        self[l[1]].dec()
                        util = True
                # digram was not found from the main sequence or from the rules
                else:
                    # append new object to the main sequence
                    main.append(digram)
                    # if character is an index, increment counter
                    if isinstance(c, RuleIndex):
                        self[c].inc()
                    # break while loop
                    break
        # if rule utility is on (as it is recommended by default), remove redundant rules
        if utilize and util:
            self.utilize()
    
    def grammar_recursive(self, rule, recursive=False):
        """ Grammar helper function """
        if not isinstance(rule, list):
            return str(rule)
        s = ''
        for i, r in enumerate(rule):
            if isinstance(r, list):
                if i == 0:
                    s += str(self.grammar_recursive(r, recursive))
                elif isinstance(r[1], RuleIndex):
                    s += "%s" % (self.grammar_recursive(self[r[1]], recursive) if recursive else RULE_INDEX_STR % r[1])
                else:
                    s += str(self.grammar_recursive(r[1], recursive))
            elif isinstance(r, RuleIndex):
                s += "%s" % (self.grammar_recursive(self[r], recursive) if recursive else RULE_INDEX_STR % r)
            else:
                s += str(r).replace("\r\n", NEWLINE_REPLACEMENT).\
                            replace("\n", NEWLINE_REPLACEMENT).\
                            replace("\r", "").\
                            replace("\t", TAB_REPLACEMENT).\
                            replace(" ", SPACE_REPLACEMENT)
        return s
    
    def grammar_sequence(self, join=False):
        """ Retrieve the main sequence / rule from the sequencer """
        x = [item[1] for item in self[0]]
        return {SEQUENCE_KEY: self.grammar_recursive(x, False) if join else x}
    
    def grammar_rules(self, join=False, recursive=False):
        """ Retrieve rest of the rules from the sequencer """
        return {x.ind(): self.grammar_recursive(x, recursive) if join else x for x in self[1:] if x}
    
    def resolve(self, flatten=True):
        """
        When sequencer has succesfully created rules from the given input, 
        resolve method can be used to decode compressed sequence back to the original input.
        Flatten argument can be used to keep/unkeep hierarchic structure present on a returned list.
        """
        def _recur(ind):
            if isinstance(ind, RuleIndex) and self[ind] is not None:
                b = []
                l = len(self[ind])-1
                for i, item in enumerate(self[ind]):
                    if item is None:
                        continue
                    if i == 0:
                        b.append(_recur(item[0]))
                        b.append(_recur(item[1]))
                    elif i == l:
                        b.append(_recur(item[1]))
                    else:
                        b.append(_recur(item[1]))
                return b
            else:
                return ind
                
        # start from main sequence / first rule
        items = [_recur(item[1]) for item in self[0]]
        # should we flatten the result?
        return flatten_list(items) if flatten else items

    def get(self):
        """ Getter for sequence """
        return list(self)
    
    def __str__(self):
        """
        String representation of the sequencer. 
        This merges only the first of the rules i.e. the main sequence
        """
        return ''.join([(RULE_INDEX_STR % i) if isinstance(i, RuleIndex) else str(i) for i in [item[1] for item in self[0]]])


class Sequencer2(list):
    """ Main class to use algorithm. This implements the array slice based approach for the algo. """
    def __init__(self, seq=[], utilize=True):
        self += [Rule([])]
        if seq:
            for c in seq:
                self.stream(c, utilize)

    def find(self, rule):
        ind, x, j = (-1, -1, -1)
        i = 0
        for x in self:
            if x:
                j = self.digram_index(x, rule)
                if j > -1:
                    x = len(x)
                    ind = i
                    break
            i += 1
        return (ind, x, j)

    def digram_index(self, target, digram):
        l = len(target)-1
        # target list length smaller than 2
        if l < 1:
            return -1
        # if target and digram are equal in length, we can compare them directly
        if l == 1:
            return 0 if target == digram else -1
        i = 0
        while i < l:
            # find "digrams" from target list and match with passed digram argument
            if target[i:i+2] == digram:
                return i
            i += 1
        return -1

    def stream(self, c, utilize = True):
        """ Main sequence handler / algorithm """
        s = self
        main = s[0]
        if len(main) < 2:
            main.append(c)
        else:
            util = False
            # loop as many times as there are no more repeating digrams
            while True:
                # create new digram
                rule = Rule(main[-1:]+[c])
                # find digram from main sequence or other rules
                ind, x, j = self.find(rule)
                # if main sequence has digram
                if ind == 0:
                    # reuse temporarily disabled index?
                    if None in s:
                        i = rule.ind(s.index(None))
                        s[i] = rule
                    else:
                        # create new unique index
                        i = rule.ind(len(s))
                        s.append(rule)
                    # increment rule counter
                    s[i].inc()
                    # replace digram left item
                    main[j] = i
                    # remove digram right item
                    del main[j+1]
                else:
                    # main sequence didnt have digram, how about other rules?
                    if ind > 0:
                        # digram is found from long rules
                        if x > 2:
                            c = rule.ind(len(s))
                            s.append(rule)
                            rule.inc()
                            # change rule content by adding new index 
                            c1 = s[ind][j+2:]
                            del s[ind][j:]
                            s[ind] += [c] + c1
                        else:
                            # lets try to retrieve index from all rules for the next digram
                            c = RuleIndex(s.index(rule))
                        # remove last item from main sequence
                        l = main.pop()
                        # if removed object is an index, decrease count
                        if isinstance(l, RuleIndex) and s[l] is not None:
                            s[l].dec()
                            util = True
                    else:
                        # append new object to the main sequence
                        main.append(c)
                        # if character is an index, increment count
                        if isinstance(c, RuleIndex):
                            s[c].inc()
                        break
            if utilize and util:
                self.utilize()
    
    def utilize(self):
        # remove redundant rules i.e. if rule is used only once on right side of the rules list
        for rule in self:
            # only rules with count = 1
            if rule is None or rule.cnt() != 1:
                continue
            self[rule.ind()] = None
            for r in self:
                # all rules except the excluded rule
                if r is None or r is rule:
                    continue
                ind = 0
                l = len(r)
                while ind < l:
                    if isinstance(r[ind], RuleIndex) and r[ind] == rule.ind():
                        c = r[ind+1:]
                        del r[ind:]
                        r += rule + c
                    ind += 1
    
    def grammar_recursive(self, rule, recursive=False):
        s = ''
        for r in rule:
            if isinstance(r, list):
                s += str(self.grammar_recursive(r, recursive))
            elif isinstance(r, RuleIndex):
                s += "%s" % (self.grammar_recursive(self[r], recursive) if recursive else RULE_INDEX_STR % r)
            else:
                s += str(r).replace("\r\n", NEWLINE_REPLACEMENT).\
                            replace("\n", NEWLINE_REPLACEMENT).\
                            replace("\r", "").\
                            replace("\t", TAB_REPLACEMENT).\
                            replace(" ", SPACE_REPLACEMENT)
        return s

    def grammar_sequence(self, join=False):
        """ Retrieve the main sequence / rule from the sequencer """
        return {SEQUENCE_KEY: self.grammar_recursive(self[0]) if join else self[0]}
    
    def grammar_rules(self, join=False, recursive=False):
        """ Retrieve rest of the rules from the sequencer """
        return {x.ind(): self.grammar_recursive(x, recursive) if join else x for x in self[1:] if x}
    
    def resolve(self, flatten=True):
        """
        When sequencer has succesfully created rules from the given input, 
        resolve method can be used to decode compressed sequence back to the original input.
        Flatten argument can be used to keep/unkeep hierarchic structure present on a returned list.
        """
        def _recur(i):
            if not isinstance(i, RuleIndex):
                return i
            return [_recur(x) for x in self[i]]

        # start from main sequence / first rule
        items = [_recur(item) for item in self[0]]
        # should we flatten the result?
        return flatten_list(items) if flatten else items

    def get(self):
        """ Getter for sequence """
        return list(self)
    
    def __str__(self):
        """
        String representation of the sequencer. 
        This merges only the first of the rules i.e. the main sequence
        """
        return ''.join([(RULE_INDEX_STR % i) if isinstance(i, RuleIndex) else str(i) for i in self[0]])


class Sequencer3():
    """
    Main class to use algorithm. 
    This implements Sequitur from the JavaScript version to Python based approach for the algo:
    https://github.com/mspandit/sequitur-python
    """
    def __init__(self, seq = None, utilize = True):
        self.first = None
        self.grammar_cache = None
        self.g = Grammar()
        self.production = self.g.root_production
        if seq:
            for c in seq:
                self.stream(c, utilize)

    def stream(self, c, utilize = True):
        self.production.last().insert_after(Symbol.factory(self.g, c))
        if self.first is None:
            self.first = True
            return
        match = self.g.get_index(self.production.last().prev)
        if not match:
            self.g.add_index(self.production.last().prev)
        elif match.next != self.production.last().prev:
            self.production.last().prev.process_match(match)

    def grammar_recursive(self, rule, recursive=False):
        s = ''
        for r in rule:
            if isinstance(r, list):
                s += str(self.grammar_recursive(r, recursive))
            elif isinstance(r, RuleIndex):
                s += "%s" % (self.grammar_recursive(self.get(True)[r], recursive) if recursive else RULE_INDEX_STR % r)
            else:
                s += str(r).replace("\r\n", NEWLINE_REPLACEMENT).\
                            replace("\n", NEWLINE_REPLACEMENT).\
                            replace("\r", "").\
                            replace("\t", TAB_REPLACEMENT).\
                            replace(" ", SPACE_REPLACEMENT)
        return s

    def grammar_sequence(self, join=False):
        """ Retrieve the main sequence / rule from the sequencer """
        x = self.get(False)[0]
        return {SEQUENCE_KEY: self.grammar_recursive(x, False) if join else x}

    def grammar_rules(self, join=False, recursive=False):
        """ Retrieve rest of the rules from the sequencer """
        rules = self.get(False)[1:]
        return {(i+1): self.grammar_recursive(x, recursive) if join else x for i, x in enumerate(rules)}

    def resolve(self, flatten=True):
        """
        When sequencer has succesfully created rules from the given input, 
        resolve method can be used to decode compressed sequence back to the original input.
        Flatten argument can be used to keep/unkeep hierarchic structure present on a returned list.
        """
        def _recur(i):
            if not isinstance(i, RuleIndex):
                return i
            return [_recur(x) for x in self.get()[i]]

        # start from main sequence / first rule
        items = [_recur(item) for item in self.get()[0]]
        # should we flatten the result?
        return flatten_list(items) if flatten else items

    def get(self, cache=True):
        if not self.grammar_cache or not cache:
            self.grammar_cache = self.g.get_grammar()
        return self.grammar_cache

    def __str__(self):
        """
        String representation of the sequencer. 
        This merges only the first of the rules i.e. the main sequence
        """
        return ''.join([(RULE_INDEX_STR % i) if isinstance(i, RuleIndex) else str(i) for i in self.get(False)[0]])

def flatten_list(items):
    """ List flattener helper function """
    for i, x in enumerate(items):
        while isinstance(items[i], list):
            items[i:i+1] = items[i]
    return items


def print_grammar(seguencer, join=True, recursive=False):
    """ Nicely output grammar of the sequencer """
    # main sequence only
    for i, item in seguencer.grammar_sequence(join).items():
        print ("%s%s" % ("%s " % i, ARROW), item)
    # rules only
    for i, item in seguencer.grammar_rules(join, recursive).items():
        print ("%s%s" % ("%s " % i, ARROW), item)
