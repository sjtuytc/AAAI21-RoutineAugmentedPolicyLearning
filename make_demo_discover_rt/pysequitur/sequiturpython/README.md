# Sequitur Python

This is a port of Sequitur from the JavaScript version to Python. Sequitur is "a method for inferring compositional hierarchies from strings" authored by Craig Nevill-Manning and Ian Witten.

See http://www.sequitur.info/ for additional details.

Original work: https://github.com/mspandit/sequitur-python

## USAGE

Download package, extract it to your working directory and change extracted directory name to: sequiturpython. Then you should be good to go.

<pre><code>
from sequiturpython.grammar import Grammar

g = Grammar()
g.train_string('abcdbcabcd')
print (g.get_grammar())
print ()
print (g.print_grammar())
</code></pre>

output:

<pre><code>
[[1, 2, 1], ['a', 2, 'd'], ['b', 'c']]

0 --(0)--> 1 2 1 
1 --(2)--> a 2 d                                         abcd
2 --(2)--> b c                                           bc
</code></pre>

## CHANGED 27.2.2016:

- Python 3 support: module paths changed to relative and print commands to print ()
- get_grammar method to get grammar in plain list format for further usage
- visual representation of the rules changed from "rule: %d" to "rule(%d)"
- other minor changes
