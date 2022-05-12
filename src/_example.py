import shapley_calculation as sv
import feature_selection as fs
import approximated_algorithms as aa

import click
import numpy as np


def create_data(n_samples):
    
    x0 = np.random.randint(0, 8, n_samples)
    x1 = np.random.randint(2, 4, n_samples)
    x3 = np.random.randint(3, 7, n_samples)
    x4 = np.random.randint(0, 6, n_samples)
    x2 = 2*x0+x1-x3
    x5 = 3*x4
    
    data = [x0, x1, x2, x3, x4, x5]
    data = np.reshape(data, (6 , n_samples))
    
    return data


def compute_shapleyvalues(_mydata, _type, _subsets_bound = -1, approx = 'max'):
    
    if _type == 'sampled':
        shapleys = aa.approximated_shapley(_mydata, approx)
    
    else:
        if _type == 'bounded':
            SVC = sv.ShapleyValueCalculator("total_correlation", "build_subsets_limited", _subsets_bound)
        elif _type == 'full':
            SVC = sv.ShapleyValueCalculator("total_correlation", "build_subsets")
        shapleys = SVC.calculate_SVs(_mydata)
        
    return shapleys

def compute_SVFR(_mydata, _type, _subsets_bound = -1, approx = 'max'):
    
    if _type == 'sampled':
        FS = aa.FeatureSelectionCalculator(_mydata, approx)
        S = FS.SVFR()
    
    else:
        if _type == 'bounded':
            SVC = sv.ShapleyValueCalculator("total_correlation", "build_subsets_limited", _subsets_bound)
            
        elif _type == 'full':
            SVC = sv.ShapleyValueCalculator("total_correlation", "build_subsets")
        
        shaps = SVC.calculate_SVs(_mydata)        
        SVFR = fs.FeatureSelectionCalculator(_mydata, SVC)
        S = SVFR.SVFR()

    return S

def compute_SVFS(_mydata, _type, _epsilon, _subsets_bound = -1, approx = 'max'):
    
    if _type == 'sampled':
        FS = aa.FeatureSelectionCalculator(_mydata, approx)
        S = FS.SVFS(_epsilon)
    
    else:
        if _type == 'bounded':
            SVC = sv.ShapleyValueCalculator("total_correlation", "build_subsets_limited", _subsets_bound)
            
        elif _type == 'full':
            SVC = sv.ShapleyValueCalculator("total_correlation", "build_subsets")
            
        SVFR = fs.FeatureSelectionCalculator(_mydata, SVC)
        S = SVFR.SVFS(_epsilon)

    return S

@click.command()
@click.option('--_algorithm', type=str)
@click.option('--_type', type=str)
@click.option('--_epsilon', type=float)
@click.option('--_subsets_bound', type=int)
@click.option('--_approx', type=int)


def main(_algorithm, _type, _epsilon = 100, _subsets_bound = -1, _approx = 'max'):
    
    _mydata = create_data(1000)
    
    shapleyvalues = compute_shapleyvalues(_mydata, _type, _subsets_bound, _approx)
    print('shapley values: ', shapleyvalues)
    
    if _algorithm == 'SVFS':
        ordering = compute_SVFS(_mydata, _type, _epsilon, _subsets_bound, _approx) 
    elif _algorithm == 'SVFR':
        ordering = compute_SVFR(_mydata, _type, _subsets_bound, _approx)

    print('ordering: ', ordering)
        

if __name__ ==  "__main__":
    main()