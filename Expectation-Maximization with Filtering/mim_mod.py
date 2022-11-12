import math
import pandas as pd
from tabulate import tabulate
import sys
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
import textwrap

# General routine for normalizing probability distributions
def normalized(d):
    rowsum = 0.0
    for k in d.keys():
        rowsum = rowsum + d[k]
    if rowsum > 0.0:
        for k in d.keys():
            d[k] = d[k] / rowsum

# General routine for converting a sequence to string
def seq2str(seq):
    string = ""
    for elem in seq:
        string += str(elem)
    return string

# General routine for sorting a dictionary by values
from operator import itemgetter
def sortbyvalue(d):
    return sorted(iter(d.items()),key = itemgetter(1), reverse=True)

# Routine for computing the G-metric (Sec. 4) between two MIM models
def gmetric(m1, m2):
    pz = m1.seqprobs()
    qz = m2.seqprobs()
    g = 0.0
    for z in pz.keys():
        if z in qz:
            g += math.sqrt(pz[z]*qz[z])
        return g

# ---------------------------------------------------------------------------------------
# Routine for printing the matrices to file
# Added by John Isaac Enriquez August 31, 2022
def print_matrix(matrix, title):
    M = pd.DataFrame.from_dict(matrix)
    M_T = M.T
    stdout_orig = sys.stdout
    with open('matrix_mod.csv', 'a') as f:
        sys.stdout = f
        print(title)
        print(tabulate(M_T, headers='keys', tablefmt='github', floatfmt='.2f'), '\n')
        sys.stdout = stdout_orig
# ---------------------------------------------------------------------------------------

class model:

    # Initialize variables

    BEGIN = 'o' # beginning symbol of all symbol sequences
    END = 'x'   # ending symbol of all symbol sequences
    x = []      # symbol sequence list
    s = []      # source sequence list
    y = dict()  # separated symbol sequences
    N = 0       # no. of elements in the sequence list i.e. len(x)
    D = []      # list of all unique symbols in symbol list
    gM = dict() # global model matrix (Eq.3)
    M = dict()  # transition matrix

    def __init__(self, x):
        
        self.x  = x
        self.N = len(self.x)
        self.D = [self.BEGIN] + sorted(set(self.x)) + [self.END]

        # Construct the Global Model Matrix
        # Initialize elements to 0.0
        for a in self.D:
            self.gM[a] = dict()
            for b in self.D:
                self.gM[a][b] = 0.0
        
        # Set elements to the no. of transitions between all symbols
        for n in range(0, self.N-1):
            a = self.x[n]
            b = self.x[n+1]
            self.gM[a][b] += 1.0
        
        # Normalize the elements
        for a in self.D:
            normalized(self.gM[a])
        
        # Print matrix to file
        file_exists = os.path.exists('matrix_mod.csv')
        if file_exists:
            os.remove('matrix_mod.csv')
        #print_matrix(self.gM, 'Global Model')

        """# Set low trasition probability to zero
        threshold = 0.1
        symbols = list(self.gM.keys())
        for a in symbols:
            for b in symbols:
                if self.gM[a][b] < threshold and self.gM[a][b] != 0.0:
                    #print('Very Low = ', a, b, self.gM[a][b])
                    self.gM[a][b] = 0
        print_matrix(self.gM, 'Low T Removed Threshold = ' + str(threshold))
      
        # Normalize  the transition matrix
        for a in self.D:
            normalized(self.gM[a])
        print_matrix(self.gM, 'Renormalized')"""

    # Print a given transition matrix T
    def printmodel(self, T):
        print((' '.ljust(5)), end=' ')
        for a in self.D:
            print(a.ljust(5), end=' ')
        print()
        for a in self.D:
            print(a.ljust(5), end=' ')
            for b in self.D:
                if T[a][b] == 0.0:
                    print('-'.ljust(5), end=' ')
                else:
                    print('{0:.2f}'.format(T[a][b]).ljust(5), end=' ')
            print()
    
    # Algorithm 1
    def estsources(self, T):
        self.s = []     # source sequence
        self.y = dict() # separated symbol sequences
        active = set()  # set of active sources
        stdout_orig = sys.stdout
        with open('matrix_mod.csv', 'a') as f:
            sys.stdout = f
            print('\n------------------------ Use x and M to calculate s and y -----------------------')
            sys.stdout = stdout_orig
        print_matrix(T, 'Input Matrix')
        for n in range(0, self.N):
            xn = self.x[n]
            pmax = 0.0
            sn = -1
            
            for k in active: # Do for all active sources
                # If new symbol is already in symbol sequence k, then
                # the symbol sequence k is not a candidate sequence.
                # This is because duplicate symbol is not allowed.
                if xn in self.y[k]: 
                    continue
            
                a = self.y[k][-1]
                b = xn
                p = T[a][b] # p is the transition probability

                if p > pmax:
                    # Assign the new symbol to the source with highest
                    # transition probability
                    sn = k
                    pmax = p

            # Make new sequence when symbol "A" is encountered
            if sn == -1 or T[self.BEGIN][xn] > pmax:
                sn = len(self.y) + 1
                active.add(sn)
                self.y[sn] = []
            
            self.s.append(sn)
            self.y[sn].append(xn)

            pnext = 0.0
            bnext = self.BEGIN

            # Remove a sequence from active sequences when the 
            # final symbol is encountered
            for b in self.D:
                if T[xn][b] > pnext:
                    pnext = T[xn][b]
                    bnext = b

            if bnext == self.END:
                active.remove(sn)
        
        # Print Algorithm 1 Result
        stdout_orig = sys.stdout
        with open('matrix_mod.csv', 'a') as f:
            sys.stdout = f
            #print('Symbol Sequence x')
            #print(textwrap.fill(str(self.x), width=80))
            print('\'---------------------------------- Result -------------------------------------')
            print('\nSource Sequence s')
            print(textwrap.fill(str(self.s), width=80))
            print('\nSeparated Symbol Sequence y')
            for i in self.y.keys():
                print(i, self.y[i])
            print('\n----------------------------------- End ---------------------------------------')
            sys.stdout = stdout_orig

    # Update the trainstition matrix based on separated symbol 
    # sequence y and sources sequence s
    def estparams(self):
        self.M = dict()

        # Initialize elements to 0.0
        for a in self.D:
            self.M[a] = dict()
            for b in self.D:
                self.M[a][b] = 0.0
        
        # Count the number of transitions between each symbol
        for k in self.y.keys():
            a = self.BEGIN
            b = self.y[k][0]
            self.M[a][b] += 1.0
            
            for r in range(0, len(self.y[k])-1):
                a = self.y[k][r]
                b = self.y[k][r+1]
                self.M[a][b] += 1.0
            
            a = self.y[k][-1]
            b = self.END
            self.M[a][b] += 1.0
        print_matrix(self.M, 'Count Symbol Occurance')

        # Normalize  the transition matrix
        for a in self.D:
            normalized(self.M[a])
        print_matrix(self.M, 'Normalized')
    
        # ---------------------------------------------------------------------------------------
        # Routine for pruning low probability transitions
        # Added by John Isaac Enriquez Sept. 1, 2022
        # Choose threshold method
        # ---------------------------------------------------------------------------------------
        '''# Method 1: Arbitrary Value
        # threshold = 0.1'''
        # ---------------------------------------------------------------------------------------
        '''# Method 2: T is D% where D is the no. of unique symbols
        threshold = 1 / len(list(self.M.keys()))
        stdout_orig = sys.stdout
        with open('matrix_mod.csv', 'a') as f:
            sys.stdout = f
            print('Threshold = ', threshold)
            sys.stdout = stdout_orig
        
        # Set low trasition probability to zero
        symbols = list(self.M.keys())
        for a in symbols:
            for b in symbols:
                if self.M[a][b] < threshold and self.M[a][b] != 0.0:
                    #print('Very Low = ', a, b, self.M[a][b])
                    self.M[a][b] = 0
        print_matrix(self.M, 'Low T Removed Threshold = ' + str(threshold))'''
        # ---------------------------------------------------------------------------------------
        # Method 3: Standard Deviation
        std_factor = 0.8
        symbols = list(self.M.keys())
        plist = []
        threshold = dict()
        for a in symbols:
            for b in symbols:
                # Make a list of symbols whose transition probability is greater than zero
                if self.M[a][b] != 0.0:
                    plist.append(self.M[a][b])
            # Calculate the mean and standard deviation of the transitions
            if len(plist) != 0:
                std = np.std(plist)
                avg = np.mean(plist)
            else: # Prevents NAN value
                std = 0.0
                avg = 0.0
            
            # Definition of threshold
            threshold[a] = avg - (std_factor * std)
            
            # Remove negative threshold values
            if threshold[a] < 0.0:
                threshold[a] = 0.0                
            plist.clear()
            #Just some codes for debuging
            #stdout_orig = sys.stdout
            #with open('matrix_mod.csv', 'a') as f:
            #    sys.stdout = f
            #    print(a)
            #    print('STD = ', std)
            #    print('Mean = ', avg)
            #    print('Threshold = ', threshold[a])
            #sys.stdout = stdout_orig
        
        stdout_orig = sys.stdout
        with open('matrix_mod.csv', 'a') as f:
            sys.stdout = f
            print('Threshold = ', threshold)
        sys.stdout = stdout_orig
        
        # Set low trasition probability to zero
        symbols = list(self.M.keys())
        for a in symbols:
            for b in symbols:
                if self.M[a][b] < threshold[a] and self.M[a][b] != 0.0:
                    print('Probabilities less than threshold = ', a, b, self.M[a][b])
                    stdout_orig = sys.stdout
                    with open('matrix_mod.csv', 'a') as f:
                        sys.stdout = f
                        print('Probabilities less than threshold = ', a, b, self.M[a][b])
                    sys.stdout = stdout_orig
                    self.M[a][b] = 0
        stdout_orig = sys.stdout
        with open('matrix_mod.csv', 'a') as f:
            sys.stdout = f
            print('Low Probability Threshold')
            print(textwrap.fill(str(threshold), width=80))
            print()
        sys.stdout = stdout_orig
        print_matrix(self.M, 'Low Probabilities Removed' + str())
        # ------------------------Threshold Implementation Ends -----------------------------------
        

        # Normalize  the transition matrix
        for a in self.D:
            normalized(self.M[a])
        print_matrix(self.M, 'Renormalized')
        stdout_orig = sys.stdout
        with open('matrix_mod.csv', 'a') as f:
            sys.stdout = f
            print('\n----------------------------------- End ---------------------------------------')
        sys.stdout = stdout_orig  

        
    # Algorithm 2
    def estimate(self):
        prevsseqs = []
        print("Initializing source sequence ...")
        stdout_orig = sys.stdout
        with open('matrix_mod.csv', 'a') as f:
            sys.stdout = f
            print('################################################################################')
            print('                                 Iteration 0 \n')
            print('Symbol Sequence x')
            print(textwrap.fill(str(self.x), width=80)+'\n')
            print('Use Global Model Matrix')
            sys.stdout = stdout_orig
        self.estsources(self.gM)
        its = 0

        while self.s not in prevsseqs:
            its += 1
            with open('matrix_mod.csv', 'a') as f:
                sys.stdout = f
                print('\n################################################################################')
                print('                                 Iteration', its, '\n')
                print('Symbol Sequence x')
                print(textwrap.fill(str(self.x), width=80))
                print('\n------------------------------ Calculate M from y -------------------------------')
            sys.stdout = stdout_orig
            print("#{0}: Estimating parameters ...".format(its))
            self.estparams()
            prevsseqs.append(self.s[:])
            
            print("#{0}: Computing source sequence ...".format(its))
            self.estsources(self.M)

            '''with open('matrix_mod.csv', 'a') as f:
                sys.stdout = f
                print('\n------------------------------ End ---------------------------------')
            sys.stdout = stdout_orig'''
    
        return len(set(self.s))
    
    # Computes the probability distribution for the different sequences 
    # produced by this model (p(z) or q(z) in the paper)
    def seqprobs(self):
        probs = dict()
        for k in self.y.keys():
            z = seq2str(self.y[k])
            if z in probs:
                probs[z] += 1.0
            else:
                probs[z] = 1.0
        normalized(probs)
        return probs
    
    # Checks that it is possible to recover the symbol sequence x from 
    # the separate sequences y (sanity check)
    def checkmodel(self):
        x2 = []
        pos = dict()
        for k in self.y:
            pos[k] = -1
        for n in range(len(self.s)):
            sn = self.s[n]
            pos[sn] += 1
            xn = self.y[sn][pos[sn]]
            x2.append(xn)
        return x2 == self.x
            



            


