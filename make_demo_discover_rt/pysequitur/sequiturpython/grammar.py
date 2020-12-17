from .symbol import Symbol
from .rule import Rule

class Grammar(object):
    """docstring for Grammar"""

    unique_rule_number = 1

    def __init__(self):
        super(Grammar, self).__init__()
        self.digram_index = {}
        self.root_production = Rule(self)

    def train_string(self, input_sequence):
        """docstring for train_string"""
        i = 0
        l = len(input_sequence)
        if i < l:
            self.root_production.last().insert_after(Symbol.factory(self, input_sequence[i]))
            i += 1
        while i < l:
            self.root_production.last().insert_after(Symbol.factory(self, input_sequence[i]))
            i += 1
            match = self.get_index(self.root_production.last().prev)
            if not match:
                self.add_index(self.root_production.last().prev)
            elif match.next != self.root_production.last().prev:
                self.root_production.last().prev.process_match(match)

    def add_index(self, digram):
        """docstring for index"""
        self.digram_index[digram.hash_value()] = digram

    def get_index(self, digram):
        """docstring for get"""
        return self.digram_index.get(digram.hash_value())

    def clear_index(self, digram):
        """docstring for clear_index"""
        if self.digram_index.get(digram.hash_value()) == digram:
            self.digram_index[digram.hash_value()] = None

    def print_grammar(self):
        """docstring for print_grammar"""
        output_array = []
        rule_set = [self.root_production]
        i = 0
        for rule in rule_set:
            output_array.append("%s --(%d)--> " % (i, rule.reference_count))
            line_length = rule.print_rule(rule_set, output_array, len("%s --(%d)--> " % (i, rule.reference_count)))
            if i > 0:
                output_array.append(' ' * (57 - line_length))
                line_length = rule.print_rule_expansion(rule_set, output_array, line_length)
            output_array.append('\n');
            i += 1
        return "".join(output_array)

    def get_grammar(self):
        """docstring for get_grammar"""
        rule_set = [self.root_production]
        return [rule.get_rule(rule_set) for rule in rule_set]
