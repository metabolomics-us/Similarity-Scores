from pandas import read_csv
import pandas as pd
import matplotlib.pyplot as plt
#from pyspec_ssmehta.parser.pymzl.msms_spectrum import MSMSSpectrum
from pyspec_ssmehta.similarity.nominal_similarity import *
from pyspec_ssmehta.similarity.nominal_similarity import _transform_spectrum_tuple
from pyspec_ssmehta.similarity.nominal_similarity import _transform_spectrum
from pyspec_ssmehta.entropy.entropy import Entropy
from pyspec_ssmehta.loader import Spectra
import numpy as np
import seaborn as sns
import time

pip install -e "git+https://github.com/metabolomics-us/pyspec.git@similarity#egg=version_subpkg&subdirectory=pyspec" --no-deps


data = read_csv('data.csv')
data.sample(frac=1)

class spectra_similarity:
    def __init__(self, filename, which_bins = None, shuffle = False):
        '''
        intensity_consensus: 
        A dictionary with bin number as the key, the value is a 2 dimension array. 
        Each row represents a consensus spectrum
        '''
        self.filename = filename
        
        self.data = read_csv(self.filename)
        if shuffle:
            self.data = self.data.sample(frac=1)
        if which_bins is None:
            self.bins = np.unique(self.data.bin_id)
        else:
            self.bins = which_bins

        data_by_bin = {}
        for i in self.bins:
            data_by_bin[i] = self.data[self.data.bin_id == i] 
            
        self.intensity = {}
        self.intensity_consensus = {}
        self.n_spectrum = {}

        for i in self.bins:
            self.n_spectrum[i] = data_by_bin[i].shape[0]
            self.intensity[i] = np.zeros([self.n_spectrum[i], (500-85+1)])
            for j in range(self.n_spectrum[i]):
                spectrum = data_by_bin[i].spectra.iloc[j]
                spectrum_tuple = _transform_spectrum_tuple(spectrum)
                for k in spectrum_tuple:
                    if k[0] <= 500:
                        self.intensity[i][j,int(k[0])-85] = k[1]

            self.intensity_consensus[i] = np.zeros([self.n_spectrum[i], (500-85+1)])
            for j in range((500-85+1)):
                self.intensity_consensus[i][:,j] = np.cumsum(self.intensity[i][:,j])

    def array_to_spec(self, array):
        spec = {}

        for i, y_val in enumerate(array):
            if y_val > 0:
                spec[85 + i] = y_val
        return spec
    
    def array_to_spec_str(self, array):
        spec_str = ''
        
        for i, j in self.array_to_spec(array).items():
            spec_str += str(i)+':'+str(j)+' '
        return spec_str.rstrip()            

    def compute_score(self,
                      ref_index = 0,
                      similarity_method = "cosine_similarity",
                      tolerance = 0.01):
        self.score = {}
        for i in self.bins:
            if ref_index < self.n_spectrum[i]:
                spec_ref = self.array_to_spec(self.intensity[i][ref_index,:])
                score_temp = []
                for j in range(self.n_spectrum[i]):
                    spec = self.array_to_spec(self.intensity_consensus[i][j,:])  
                    if similarity_method == "spectral_similarity":        
                        spec_ref_MSMS = MSMSSpectrum(spec_ref, precursor_mz = self.intensity_consensus[i][0,0])
                        spec_MSMS = MSMSSpectrum(spec, precursor_mz = self.intensity_consensus[i][j,0])
                        score_temp.append(spec_ref_MSMS.spectral_similarity(spec_MSMS, tolerance))
                    elif similarity_method == 'cosine_similarity':
                        score_temp.append(cosine_similarity(spec_ref, spec))
                    elif similarity_method == 'composite_similarity':
                        score_temp.append(composite_similarity(spec_ref, spec))
                self.score[i] = np.array(score_temp)
            else: 
                self.score[i] = np.zeros((self.n_spectrum[i],))

    def compute_entropy(self):
        self.entropy = {}
        for i in self.bins:
            entropy_temp = []
            for j in range(self.n_spectrum[i]):
                spec_str = self.array_to_spec_str(self.intensity_consensus[i][j,:]) 
                spectra = Spectra(ms_level=1,spectra=spec_str)                   
                entropy_temp.append(Entropy().compute(spectra)[0])
            self.entropy[i] = np.array(entropy_temp)
                
    def cross_ref(self, 
                  similarity_method = "cosine_similarity ", 
                  tolerance = 0.01):
        self.score_matrix = {}
        
        n_spectrum_max = np.max(list(spectra_out.n_spectrum.values()))
        score_arrays_collection = {}
        
        for j in range(n_spectrum_max):
            self.compute_score(ref_index = j)
            for i in self.bins:
                if j == 0:
                    score_arrays_collection[i] = []
                if j < self.n_spectrum[i]:
                    score_arrays_collection[i].append(self.score[i])
        
        for i in self.bins:
            self.score_matrix[i] = np.vstack(score_arrays_collection[i])

        self.score_mid_val = {}            
        self.score_median = {}
        self.score_sd = {}  
        self.score_mean = {}
        
        for i in self.bins:   
            max_val = np.max(self.score_matrix[i], axis = 0)
            min_val = np.min(self.score_matrix[i], axis = 0)
            mid_val = (max_val+min_val)/2
            self.score_mid_val[i] = mid_val
            self.score_median[i] = np.median(self.score_matrix[i], axis = 0)
            self.score_sd[i] = np.std(self.score_matrix[i], axis = 0)
            self.score_mean[i] = np.mean(self.score_matrix[i], axis = 0)

    def similarity_plot(self, metric = "mean"):
        if metric == "mid_val":
            for i in self.bins:
                x = list(range(1,self.n_spectrum[i]+1))
                plt.plot(x, self.score_mid_val[i], label=f'bin: {i}')
                plt.xlabel('Count of spectra in the consensus spectra') 
                plt.ylabel('Score') # y axis is score
                plt.ylim((0.8,1))
                plt.title('Mid-Value')
                plt.legend()
                plt.rc('font', size=15) 
                figure_name = 'mid_val_score_bin_' + str(i) + '.png'
                plt.savefig(figure_name, bbox_inches='tight')
                plt.show()    
                plt.close()
        elif metric == "median":
            for i in self.bins:
                x = list(range(1,self.n_spectrum[i]+1))
                plt.plot(x, self.score_median[i], label=f'bin: {i}')
                plt.xlabel('Count of spectra in the consensus spectra') 
                plt.ylabel('Score') # y axis is score     
                plt.ylim((0.8,1))                
                plt.title('Median')
                plt.legend()
                plt.rc('font', size=15) 
                figure_name = 'median_score_bin_' + str(i) + '.png'
                plt.savefig(figure_name, bbox_inches='tight')
                plt.show()    
                plt.close()
        elif metric == "std":
            for i in self.bins:
                x = list(range(1,self.n_spectrum[i]+1))
                plt.plot(x, self.score_sd[i], label=f'bin: {i}')
                plt.xlabel('Count of spectra in the consensus spectra') 
                plt.ylabel('Score') # y axis is score
                plt.title('Standard Deviation')
                plt.legend()
                plt.rc('font', size=15) 
                figure_name = 'std_score_bin_' + str(i) + '.png'
                plt.savefig(figure_name, bbox_inches='tight')
                plt.show()    
                plt.close()
        elif metric == "mean":
            for i in self.bins:
                x = list(range(1,self.n_spectrum[i]+1))
                plt.plot(x, self.score_mean[i], label=f'bin: {i}')
                plt.xlabel('Count of spectra in the consensus spectra') 
                plt.ylabel('Score') # y axis is score
                plt.ylim((0.8,1))                
                plt.title('Mean for Bin ' + str(i))
                #plt.legend()
                plt.rc('font', size=15) 
                figure_name = 'mean_score_bin_' + str(i) + '.png'
                plt.savefig(figure_name, bbox_inches='tight')
                plt.show()    
                plt.close()

    def plot_metrics(self, selected_bin = '18,223'):
        plt.rc('font', size=15) 
        plt.figure(figsize=(20,10))
        x = list(range(1,self.n_spectrum[selected_bin]+1))
        for j in range(len(x)):
            plt.plot(x, self.score_matrix[selected_bin][j], 'lightgrey')
        plt.plot(x, self.score[selected_bin], 'brown', label='mid value') 
        plt.plot(x, self.score_median[selected_bin], 'r', label='median')   
        plt.plot(x, self.score_mean[selected_bin], 'b', label='mean') 
        plt.plot(x, self.score_sd[selected_bin], 'g', label='standard devation') 
        plt.xlabel('Count of spectra in the consensus spectra') 
        plt.ylabel('Score') # y axis is score
        plt.title(f'Similarity Curves with different/
                  reference spectrum for bin: {selected_bin}')
        plt.legend()
        plt.show()

    def plot_entropy(self, selected_bin = '18,223'):
        plt.rc('font', size=15) 
        plt.figure(figsize=(20,10))
        x = list(range(1,self.n_spectrum[selected_bin]+1))
        plt.plot(x, self.entropy[selected_bin], 'black') 
        plt.xlabel('Count of spectra in the consensus spectra') 
        plt.ylabel('Entropy') # y axis is score
        plt.title(f'Entropy Curves for bin: {selected_bin}')
        plt.show() 


# To test the code --> 
spectra_out = spectra_similarity(filename = 'result.csv', shuffle = True)

spectra_out.cross_ref()

spectra_out.score
spectra_out.similarity_plot()  
# get all the plots  
spectra_out.plot_metrics()


spectra_out.similarity_plot() # mean graph
spectra_out.similarity_plot(metric = "median")
spectra_out.similarity_plot(metric = "std")
spectra_out.similarity_plot(metric ="mid_val")

