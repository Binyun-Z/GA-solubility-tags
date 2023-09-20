import os, sys, re
import json
import random
import math
import cmath
import pickle
import itertools
import numpy as np
import pandas as pd
import warnings
from collections import Counter
from sklearn.cluster import (KMeans, AffinityPropagation, MeanShift, estimate_bandwidth, DBSCAN,
                             AgglomerativeClustering, SpectralClustering, MiniBatchKMeans)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.manifold import TSNE
from rdkit import Chem
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import copy

plt.style.use('ggplot')
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class Sequence(object):
    def __init__(self, file):
        self.file = file                              # whole file path
        self.fasta_list = []                          # 2-D list [sampleName, fragment, label, training or testing]
        self.sequence_number = 0                      # int: the number of samples
        self.is_equal = False                         # bool: sequence with equal length?
        self.minimum_length = 1                       # int
        self.maximum_length = 0                       # int
        self.minimum_length_without_minus = 1         # int
        self.maximum_length_without_minus = 0         # int

        if(isinstance(file,list)):
            self.fasta_list,self.msg = self.read_list(self.file)
        else:
            self.fasta_list,self.msg = self.read_fasta(self.file)
        self.sequence_number = len(self.fasta_list)

        if self.sequence_number > 0:
            self.is_equal, self.minimum_length, self.maximum_length, self.minimum_length_without_minus, self.maximum_length_without_minus = self.sequence_with_equal_length()

        else:
            self.error_msg = 'File format error.'
    def read_list(self,seqlist):
        """
        load fasta sequence from list
        :param list:
        :return: fasta_sequences, msg
        """
        msg = ''
        if len(seqlist)==0:
            msg = 'sequences does not exist.'
            return [], msg

        fasta_sequences = []
        for i,seq in  enumerate(seqlist):
            name = "pet"+str(i)
            label=None
            label_train = None
            fasta_sequences.append([name, seq, label, label_train])
        return fasta_sequences, msg

    def read_fasta(self, file):
        """
        read fasta sequence
        :param file:
        :return:
        """
        msg = ''
        if not os.path.exists(self.file):
            msg = 'Error: file %s does not exist.' % self.file
            return [], msg
        with open(file) as f:
            records = f.read()
        records = records.split('>')[1:]
        fasta_sequences = []
        for fasta in records:
            array = fasta.split('\n')
            header, sequence = array[0].split()[0], re.sub('[^ACDEFGHIKLMNPQRSTUVWY-]', '-', ''.join(array[1:]).upper())
            header_array = header.split('|')
            name = header_array[0]
            label = header_array[1] if len(header_array) >= 2 else '0'
            label_train = header_array[2] if len(header_array) >= 3 else 'training'
            fasta_sequences.append([name, sequence, label, label_train])
        return fasta_sequences, msg

    def sequence_with_equal_length(self):
        """
        Check if fasta sequence is in equal length
        :return:
        """
        length_set = set()
        length_set_1 = set()
        for item in self.fasta_list:
            length_set.add(len(item[1]))
            length_set_1.add(len(re.sub('-', '', item[1])))

        length_set = sorted(length_set)
        length_set_1 = sorted(length_set_1)
        if len(length_set) == 1:
            return True, length_set[0], length_set[-1], length_set_1[0], length_set_1[-1]
        else:
            return False, length_set[0], length_set[-1], length_set_1[0], length_set_1[-1]

    def check_sequence_type(self):
        """
        Specify sequence type (Protein, DNA or RNA)
        :return:
        """
        tmp_fasta_list = []
        if len(self.fasta_list) < 100:
            tmp_fasta_list = self.fasta_list
        else:
            random_index = random.sample(range(0, len(self.fasta_list)), 100)
            for i in random_index:
                tmp_fasta_list.append(self.fasta_list[i])

        sequence = ''
        for item in tmp_fasta_list:
            sequence += item[1]

        char_set = set(sequence)
        if 5 < len(char_set) <= 21:
            for line in self.fasta_list:
                line[1] = re.sub('[^ACDEFGHIKLMNPQRSTVWY]', '-', line[1])
            return 'Protein'
        elif 0 < len(char_set) <= 5 and 'T' in char_set:
            return 'DNA'
        elif 0 < len(char_set) <= 5 and 'U' in char_set:
            for line in self.fasta_list:
                line[1] = re.sub('U', 'T', line[1])
            return 'RNA'
        else:
            return 'Unknown'

class iProtein(Sequence):
    """
    # Running examples:
    # import iFeatureOmegaCLI
    >>> import iFeatureOmegaCLI

    # create a instance
    >>> protein = iFeatureOmegaCLI.iProtein("./data_examples/peptide_sequences.txt")

    # display available feature descriptor methods
    >>> protein.display_feature_types()

    # import parameters for feature descriptors (optimal)
    >>> protein.import_parameters('parameters/Protein_parameters_setting.json')

    # calculate feature descriptors. Take "AAC" as an example.
    >>> protein.get_descriptor("AAC")

    # display the feature descriptors
    >>> print(protein.encodings)

    # save feature descriptors
    >>> protein.to_csv("AAC.csv", "index=False", header=False)
    """

    def __init__(self, file):
        super(iProtein, self).__init__(file=file)
        self.__default_para_dict = {
            'EAAC': {'sliding_window': 5},
            'CKSAAP type 1': {'kspace': 3},
            'CKSAAP type 2': {'kspace': 3},
            'EGAAC': {'sliding_window': 5},
            'CKSAAGP type 1': {'kspace': 3},
            'CKSAAGP type 2': {'kspace': 3},
            'AAIndex': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101'},
            'NMBroto': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3,},
            'Moran': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3,},
            'Geary': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3,},
            'KSCTriad': {'kspace': 3},
            'SOCNumber': {'nlag': 3},
            'QSOrder': {'nlag': 3, 'weight': 0.05},
            'PAAC': {'weight': 0.05, 'lambdaValue': 3},
            'APAAC': {'weight': 0.05, 'lambdaValue': 3},
            'DistancePair': {'distance': 0, 'cp': 'cp(20)',},
            'AC': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3},
            'CC': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3},
            'ACC': {'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 'nlag': 3},
            'PseKRAAC type 1': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 2},
            'PseKRAAC type 2': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 2},
            'PseKRAAC type 3A': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 2},
            'PseKRAAC type 3B': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 2},
            'PseKRAAC type 4': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 5},
            'PseKRAAC type 5': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 3},
            'PseKRAAC type 6A': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 4},
            'PseKRAAC type 6B': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 5},
            'PseKRAAC type 6C': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 5},
            'PseKRAAC type 7': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 2},
            'PseKRAAC type 8': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 2},
            'PseKRAAC type 9': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 2},
            'PseKRAAC type 10': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 2},
            'PseKRAAC type 11': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 2},
            'PseKRAAC type 12': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 2},
            'PseKRAAC type 13': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 4},
            'PseKRAAC type 14': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 2},
            'PseKRAAC type 15': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 2},
            'PseKRAAC type 16': {'lambdaValue': 3, 'PseKRAAC_model': 'g-gap', 'g-gap': 2, 'k-tuple': 2, 'RAAC_clust': 2},            
        }
        self.__default_para = {
            'sliding_window': 5,
            'kspace': 3,            
            'nlag': 3,
            'weight': 0.05,
            'lambdaValue': 3,
            'PseKRAAC_model': 'g-gap',
            'g-gap': 2,
            'k-tuple': 2,
            'RAAC_clust': 1,
            'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101', 
        }
        self.encodings = None       # pandas dataframe
        self.__cmd_dict ={
            'AAC': 'self._AAC()',
            'EAAC': 'self._EAAC()',
            'CKSAAP type 1': 'self._CKSAAP(normalized=True)',
            'CKSAAP type 2': 'self._CKSAAP(normalized=False)',
            'DPC type 1': 'self._DPC(normalized=True)',
            'DPC type 2': 'self._DPC(normalized=False)',
            'DDE': 'self._DDE()',
            'TPC type 1': 'self._TPC(normalized=True)',
            'TPC type 2': 'self._TPC(normalized=False)',
            'binary': 'self._binary()',
            'binary_6bit': 'self._binary_6bit()',
            'binary_5bit type 1': 'self._binary_5bit_type_1()',
            'binary_5bit type 2': 'self._binary_5bit_type_2()',
            'binary_3bit type 1': 'self._binary_3bit_type_1()',
            'binary_3bit type 2': 'self._binary_3bit_type_2()',
            'binary_3bit type 3': 'self._binary_3bit_type_3()',
            'binary_3bit type 4': 'self._binary_3bit_type_4()',
            'binary_3bit type 5': 'self._binary_3bit_type_5()',
            'binary_3bit type 6': 'self._binary_3bit_type_6()',
            'binary_3bit type 7': 'self._binary_3bit_type_7()',
            'AESNN3': 'self._AESNN3()',
            'GAAC': 'self._GAAC()',
            'EGAAC': 'self._EGAAC()',
            'CKSAAGP type 1': 'self._CKSAAGP(normalized=True)',
            'CKSAAGP type 2': 'self._CKSAAGP(normalized=False)',
            'GDPC type 1': 'self._GDPC(normalized=True)',
            'GDPC type 2': 'self._GDPC(normalized=False)',
            'GTPC type 1': 'self._GTPC(normalized=True)',
            'GTPC type 2': 'self._GTPC(normalized=False)',
            'AAIndex': 'self._AAIndex()',
            'ZScale': 'self._ZScale()',
            'BLOSUM62': 'self._BLOSUM62()',
            'NMBroto': 'self._NMBroto()',
            'Moran': 'self._Moran()',
            'Geary': 'self._Geary()',
            'CTDC': 'self._CTDC()',
            'CTDT': 'self._CTDT()',
            'CTDD': 'self._CTDD()',
            'CTriad': 'self._CTriad()',
            'KSCTriad': 'self._KSCTriad()',
            'SOCNumber': 'self._SOCNumber()',
            'QSOrder': 'self._QSOrder()',
            'PAAC': 'self._PAAC()',
            'APAAC': 'self._APAAC()',
            'OPF_10bit': 'self._OPF_10bit()',
            'OPF_10bit type 1': 'self._OPF_10bit_type_1()',
            'OPF_7bit type 1': 'self._OPF_7bit_type_1()',
            'OPF_7bit type 2': 'self._OPF_7bit_type_2()',
            'OPF_7bit type 3': 'self._OPF_7bit_type_3()',
            'ASDC': 'self._ASDC()',
            'DistancePair': 'self._DistancePair()',
            'AC': 'self._AC()',
            'CC': 'self._CC()',
            'ACC': 'self._ACC()',
            'PseKRAAC type 1': 'self._PseKRAAC_type_1()',
            'PseKRAAC type 2': 'self._PseKRAAC_type_2()',
            'PseKRAAC type 3A': 'self._PseKRAAC_type_3A()',
            'PseKRAAC type 3B': 'self._PseKRAAC_type_3B()',
            'PseKRAAC type 4': 'self._PseKRAAC_type_4()',
            'PseKRAAC type 5': 'self._PseKRAAC_type_5()',
            'PseKRAAC type 6A': 'self._PseKRAAC_type_6A()',
            'PseKRAAC type 6B': 'self._PseKRAAC_type_6B()',
            'PseKRAAC type 6C': 'self._PseKRAAC_type_6C()',
            'PseKRAAC type 7': 'self._PseKRAAC_type_7()',
            'PseKRAAC type 8': 'self._PseKRAAC_type_8()',
            'PseKRAAC type 9': 'self._PseKRAAC_type_9()',
            'PseKRAAC type 10': 'self._PseKRAAC_type_10()',
            'PseKRAAC type 11': 'self._PseKRAAC_type_11()',
            'PseKRAAC type 12': 'self._PseKRAAC_type_12()',
            'PseKRAAC type 13': 'self._PseKRAAC_type_13()',
            'PseKRAAC type 14': 'self._PseKRAAC_type_14()',
            'PseKRAAC type 15': 'self._PseKRAAC_type_15()',
            'PseKRAAC type 16': 'self._PseKRAAC_type_16()',
            'KNN': 'self._KNN()',
        }

    def import_parameters(self, file):
        if os.path.exists(file):
            with open(file) as f:
                records = f.read().strip()
            try:
                self.__default_para_dict = json.loads(records)
                print('File imported successfully.')
            except Exception as e:
                print('Parameter file parser error.')

    def get_descriptor(self, descriptor='AAC'):
        # copy parameters
        if descriptor in self.__default_para_dict:
            for key in self.__default_para_dict[descriptor]:
                self.__default_para[key] = self.__default_para_dict[descriptor][key]       
            
        if descriptor in self.__cmd_dict:
            cmd = self.__cmd_dict[descriptor]
            status = eval(cmd)            
        else:
            print('The descriptor type does not exist.')

    def display_feature_types(self):
        info = '''
        ----- Available feature types ------        
        
        AAC                                                Amino acid composition
        EAAC                                               Enhanced amino acid composition
        CKSAAP type 1                                      Composition of k-spaced amino acid pairs type 1 - normalized
        CKSAAP type 2                                      Composition of k-spaced amino acid pairs type 2 - raw count
        DPC type 1                                         Dipeptide composition type 1 - normalized
        DPC type 2                                         Dipeptide composition type 2 - raw count
        TPC type 1                                         Tripeptide composition type 1 - normalized
        TPC type 2                                         Tripeptide composition type 1 - raw count
        CTDC                                               Composition
        CTDT                                               Transition
        CTDD                                               Distribution
        CTriad                                             Conjoint triad
        KSCTriad                                           Conjoint k-spaced triad
        ASDC                                               Adaptive skip dipeptide composition
        DistancePair                                       PseAAC of distance-pairs and reduced alphabe
        GAAC                                               Grouped amino acid composition
        EGAAC                                              Enhanced grouped amino acid composition
        CKSAAGP type 1                                     Composition of k-spaced amino acid group pairs type 1- normalized
        CKSAAGP type 2                                     Composition of k-spaced amino acid group pairs type 2- raw count
        GDPC type 1                                        Grouped dipeptide composition type 1 - normalized
        GDPC type 2                                        Grouped dipeptide composition type 2 - raw count
        GTPC type 1                                        Grouped tripeptide composition type 1 - normalized
        GTPC type 2                                        Grouped tripeptide composition type 1 - raw count
        Moran                                              Moran
        Geary                                              Geary
        NMBroto                                            Normalized Moreau-Broto
        AC                                                 Auto covariance
        CC                                                 Cross covariance
        ACC                                                Auto-cross covariance
        SOCNumber                                          Sequence-order-coupling number
        QSOrder                                            Quasi-sequence-order descriptors
        PAAC                                               Pseudo-amino acid composition
        APAAC                                              Amphiphilic PAAC
        PseKRAAC type 1                                    Pseudo K-tuple reduced amino acids composition type 1
        PseKRAAC type 2                                    Pseudo K-tuple reduced amino acids composition type 2
        PseKRAAC type 3A                                   Pseudo K-tuple reduced amino acids composition type 3A
        PseKRAAC type 3B                                   Pseudo K-tuple reduced amino acids composition type 3B
        PseKRAAC type 4                                    Pseudo K-tuple reduced amino acids composition type 4
        PseKRAAC type 5                                    Pseudo K-tuple reduced amino acids composition type 5
        PseKRAAC type 6A                                   Pseudo K-tuple reduced amino acids composition type 6A
        PseKRAAC type 6B                                   Pseudo K-tuple reduced amino acids composition type 6B
        PseKRAAC type 6C                                   Pseudo K-tuple reduced amino acids composition type 6C
        PseKRAAC type 7                                    Pseudo K-tuple reduced amino acids composition type 7
        PseKRAAC type 8                                    Pseudo K-tuple reduced amino acids composition type 8
        PseKRAAC type 9                                    Pseudo K-tuple reduced amino acids composition type 9
        PseKRAAC type 10                                   Pseudo K-tuple reduced amino acids composition type 10
        PseKRAAC type 11                                   Pseudo K-tuple reduced amino acids composition type 11
        PseKRAAC type 12                                   Pseudo K-tuple reduced amino acids composition type 12
        PseKRAAC type 13                                   Pseudo K-tuple reduced amino acids composition type 13
        PseKRAAC type 14                                   Pseudo K-tuple reduced amino acids composition type 14
        PseKRAAC type 15                                   Pseudo K-tuple reduced amino acids composition type 15
        PseKRAAC type 16                                   Pseudo K-tuple reduced amino acids composition type 16
        binary                                             Binary
        binary_6bit                                        Binary
        binary_5bit type 1                                 Binary
        binary_5bit type 2                                 Binary
        binary_3bit type 1                                 Binary
        binary_3bit type 2                                 Binary
        binary_3bit type 3                                 Binary
        binary_3bit type 4                                 Binary
        binary_3bit type 5                                 Binary
        binary_3bit type 6                                 Binary
        binary_3bit type 7                                 Binary
        AESNN3                                             Learn from alignments
        OPF_10bit                                          Overlapping property features - 10 bit
        OPF_7bit type 1                                    Overlapping property features - 7 bit type 1
        OPF_7bit type 2                                    Overlapping property features - 7 bit type 2
        OPF_7bit type 3                                    Overlapping property features - 7 bit type 3
        AAIndex                                            AAIndex
        BLOSUM62                                           BLOSUM62
        ZScale                                             Z-Scales index
        KNN                                                K-nearest neighbor

        Note: the first column is the names of availables feature types while the second column is description.  
        
        '''

        print(info)

    def add_samples_label(self, file):
        with open(file) as f:
            labels = f.read().strip().split('\n')        
        for i in range(np.min([len(self.fasta_list), len(labels)])):
            self.fasta_list[i][2] = '1' if labels[i] == '1' else '0'

    def _AAC(self):
        try:
            AA = 'ACDEFGHIKLMNPQRSTVWY'
            header = ['SampleName']
            encodings = []
            for i in AA:
                header.append('AAC_{0}'.format(i))
            encodings.append(header)
            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                count = Counter(sequence)
                for key in count:
                    count[key] = count[key] / len(sequence)
                code = [name]
                for aa in AA:
                    code.append(count[aa])
                encodings.append(code)
            encodings = np.array(encodings)
            self.encodings = pd.DataFrame(encodings[1:, 1:].astype(float), columns=encodings[0, 1:], index=encodings[1:, 0])
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False
    
    def to_csv(self, file="encode.csv", index=False, header=False):
        try:
            self.encodings.to_csv(file, index=index, header=header)
        except Exception as e:
            self.error_msg = str(e)
            return False
        return True
    
    def to_tsv(self, file="encode.tsv", index=False, header=False):
        try:
            self.encodings.to_csv(file, sep='\t', index=index, header=header)
        except Exception as e:
            self.error_msg = str(e)
            return False
        return True

    def to_svm(self, file="encode.svm"):
        try:
            with open(file, 'w') as f:
                for line in self.encodings.values:
                    f.write('1')
                    for i in range(len(line)):
                        f.write('  %d:%s' % (i+1, line[i]))
                    f.write('\n')
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def to_arff(self, file="encode.arff"):
        with open(file, 'w') as f:
            f.write('@relation descriptor\n\n')
            for i in range(1, len(self.encodings.values[0]) + 1):
                f.write('@attribute f.%d numeric\n' % i)
            f.write('@attribute play {yes, no}\n\n')
            f.write('@data\n')
            for line in self.encodings.values:
                line = line
                for fea in line:
                    f.write('%s,' % fea)                
                f.write('yes\n')