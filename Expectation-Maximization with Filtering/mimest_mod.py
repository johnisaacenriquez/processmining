
import json
import sys
import mim_mod


# read symbol sequence x from stdin, with one symbol per line
x = []
for line in sys.stdin:
	symbol = line.strip()
	if len(symbol) > 0:
		x += [symbol]

# print the sequence as string
print("Symbol sequence: ", mim_mod.seq2str(x))

print("({0} symbols)".format(len(x)))

# create to be estimated from sequence x
m = mim_mod.model(x)

# estimate model
K = m.estimate()
print(K)

# print model
m.printmodel(m.M)

# show the probability distribution of the different sequences in the model
pz = mim_mod.sortbyvalue(m.seqprobs())
with open('sources.txt', 'w') as f:
	for i, (z, p) in enumerate(pz):
		print('{0:.3f} : {1}'.format(p, z))
		print('{0:.3f}@{1}'.format(p, z), file=f)

print('Total number of sources: {0}'.format(K))

# Sanity Check
S = m.checkmodel()
if S == True:
	print('Model Sanity Check Passed')
