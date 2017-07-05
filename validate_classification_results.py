# Compare the curated results to jason's results
import my
import pandas
import numpy as np
import os
import scipy.optimize
import argparse

## Parse arguments
parser = argparse.ArgumentParser(description="""
    Run the kalman filter on an example whiskers dataset.
""")
parser.add_argument("results",
    help='Name of the results file containing labeled whiskers')
args = parser.parse_args()
results_filename = args.results

## Load curated results
curated_filename = os.path.expanduser(
    '~/mnt/nas2_home/whisker/test_bed/161215_KM91/curated')
curated_df = pandas.read_pickle(curated_filename)

# Load names of whiskers
curated_num2name_filename = curated_filename + '_num2name'
curated_num2name = pandas.read_table(curated_filename + '_num2name', sep=' ', 
    names=['color_group', 'whisker']).set_index('color_group')['whisker']

## Load jason's results
results = my.misc.pickle_load(results_filename)

# Make labels consistent
jres = results[['frame', 'seg', 'color_group']].set_index(['frame', 'seg'])
cres = curated_df.set_index(['frame', 'seg'])

## Join jason's results onto curated results
ares = cres.join(jres, rsuffix='_jason', how='inner')

# For right now drop the unlabeled ones in the curated dataset
ares = ares[ares.color_group != -1].copy()

## Print general statistics of each dataset
for setname, label_set in zip(['Test', 'Curated', 'Joint'], [jres, cres, ares]):
    print "%s dataset:\n----" % setname
    unique_labels = label_set['color_group'].value_counts().index.values
    print "%d groups: %s (in order of prevalence)" % (
        len(unique_labels), ','.join(map(str, unique_labels)))
    print "%d rows, of which %d unlabeled (-1)" % (
        len(label_set), np.sum(label_set['color_group'] == -1))
    print

## Figure out the mapping between j_labels and c_labels
# Count confusion matrix
confusion_matrix = ares.reset_index().pivot_table(
    index='color_group', columns='color_group_jason', values='frame', 
    aggfunc='count').fillna(0).astype(np.int)

# Assign
# j_labels will be Jason's labels for curated classes in sorted order
# j_labels_idx is an index into the actual labels, which are on the pandas columns
c_labels_idx, j_labels_idx = scipy.optimize.linear_sum_assignment(-confusion_matrix.values)
j_labels = confusion_matrix.columns.values[j_labels_idx]
c_labels = confusion_matrix.index.values[c_labels_idx]
unused_j_labels = np.array([j_label for j_label in confusion_matrix.columns 
    if j_label not in j_labels])

# Sort the columns of the confusion matrix to match c_labels
new_column_order = list(j_labels) + list(unused_j_labels)
confusion_matrix = confusion_matrix.loc[:, new_column_order]

# Print results
print "Assignments (C>J):\n%s" % ('\n'.join(['%s (%s) > %s' % (c_label, curated_num2name[c_label], j_label) 
    for c_label, j_label in zip(c_labels, j_labels)]))
print "Unassigned labels: %s" % (' '.join(map(str, unused_j_labels)))
print

## Score each frame
# First translate j_labels to corresponding c_labels
ares['jason_translated'] = ares['color_group_jason'].replace(
    to_replace=j_labels, value=c_labels)

# Mark ones that don't correspond to any c_label as nan
ares.loc[~ares['color_group_jason'].isin(j_labels), 'jason_translated'] = np.nan

## Calculate performance
perf_whisker_l = []
n_hits = 0
for c_label, j_label in zip(c_labels, j_labels):
    n_hits += confusion_matrix.loc[c_label, j_label]
    perf_whisker = confusion_matrix.loc[c_label, j_label] / float(
        confusion_matrix.loc[c_label].sum())
    perf_whisker_l.append(perf_whisker)
frac_correct = n_hits / float(len(ares))    
worst_whisker = np.min(perf_whisker_l)

print "Overall: %0.2f" % frac_correct
print "Worst whisker: %0.2f" % worst_whisker
print "Confusion matrix: "
relabeled_confusion_matrix = confusion_matrix.join(curated_num2name).set_index(
    'whisker')
print relabeled_confusion_matrix
print

## Metrics
sensitivity = (relabeled_confusion_matrix.values.diagonal() / 
    relabeled_confusion_matrix.sum(1))

# This will fail if there are unused j_labels
specificity = (relabeled_confusion_matrix.values.diagonal() / 
    relabeled_confusion_matrix.sum(0))
specificity.index = sensitivity.index
metrics = pandas.concat([sensitivity, specificity], axis=1, verify_integrity=True,
    keys=['sensitivity', 'specificity'])
print metrics