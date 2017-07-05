# Compare the curated results to jason's results
import my
import pandas
import numpy as np
import os
import scipy.optimize
import argparse
import whiskvid
import tables
import matplotlib.pyplot as plt

## Parse arguments
parser = argparse.ArgumentParser(description="""
    Run the kalman filter on an example whiskers dataset.
""")
parser.add_argument("results",
    help='Name of the results file containing labeled whiskers')
args = parser.parse_args()
results_filename = args.results

## Load curated results
# curated_filename = os.path.expanduser(
#     '~/mnt/nas2_home/whisker/test_bed/161215_KM91/curated')
curated_filename = os.path.expanduser(
    '/mnt/nas2/homes/chris/whisker/test_bed/161215_KM91'
    )
curated_df = pandas.read_pickle(curated_filename)

# Load names of whiskers
curated_num2name_filename = curated_filename + '_num2name'
curated_num2name = pandas.read_table(curated_filename + '_num2name', sep=' ', 
    names=['color_group', 'whisker']).set_index('color_group')['whisker']

# hand chosen
curated_colors = ['b', 'g', 'r', 'purple', 'lime', 'pink', 'orange', 'white']

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

## Calculate performance
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

## Score each frame
# First translate j_labels to corresponding c_labels
ares['jason_translated'] = ares['color_group_jason'].replace(
    to_replace=j_labels, value=c_labels)

# Translate unassigned jlabels as np.nan
ares.loc[~ares['color_group_jason'].isin(j_labels), 'jason_translated'] = np.nan
ares['correct'] = ares['color_group'] == ares['jason_translated']
score_by_frame = ares['correct'].sum(level=0) / ares['correct'].count(level=0)

## Find frames that are maximally correct
n_example_frames_per_type = 5

# Find frames that are at least as good as the top N frames
best_frames = score_by_frame.sort_values(ascending=False)[
    :n_example_frames_per_type]
best_frames = score_by_frame[
    score_by_frame >= best_frames.min()]

# Choose a subset of them
example_best_frames = my.misc.take_equally_spaced(np.sort(best_frames.index), 
    n_example_frames_per_type)

## Identify the most frequent error types
confusion_matrix_errors_only = confusion_matrix.copy()
for idx in range(len(confusion_matrix_errors_only.index)):
    confusion_matrix_errors_only.iloc[idx, idx] = 0
error_types_df = confusion_matrix_errors_only.unstack().sort_values(
    ascending=False)

# Find frames corresponding to each error type
n_error_types_to_plot = 3
grouped_ares = ares.groupby(error_types_df.index.names)
example_frames_l = []
error_types_to_plot = error_types_df.index[:n_error_types_to_plot]
for group_key in error_types_to_plot:
    # Get all frames with this error
    error_frames = grouped_ares.get_group(group_key).index.get_level_values('frame')
    
    # Choose a subset
    if len(error_frames) <= n_example_frames_per_type:
        example_frames = error_frames
    else:
        example_frames = my.misc.take_equally_spaced(error_frames, 
            n_example_frames_per_type)
    
    example_frames_l.append(example_frames)

## Plot example frames
all_example_frames = np.concatenate([example_best_frames] + example_frames_l)
video_session = whiskvid.django_db.VideoSession.from_name('161215_KM91')
contacts_table = video_session.data.clustered_tac.load_data()

# Insert 
cwe = video_session.data.colorized_whisker_ends.load_data()
cwe_thisdata = cwe.drop('color_group', 1).join(
    ares[['color_group', 'color_group_jason']],
    on=['frame', 'seg'], how='inner')
cwe_cdata = cwe_thisdata.drop('color_group_jason', 1)
cwe_jdata = cwe_thisdata.drop('color_group', 1).rename(
    columns={'color_group_jason': 'color_group'})

# Define the test set colors
test_set_colors = []
for test_idx in range(np.max(j_labels) + 1):
    if test_idx in j_labels:
        # This will fail if c_labels are not a simple list from 0 to N-1
        c_label = c_labels[list(j_labels).index(test_idx)]
        test_set_colors.append(curated_colors[c_label])
    else:
        test_set_colors.append('k')

for n_error_type, example_frames_to_plot in enumerate(example_frames_l + 
    [example_best_frames]):
    # Create figure handles
    figure_handle, axa = plt.subplots(2, len(example_frames_to_plot), 
        figsize=(10, 4))
    figure_handle.subplots_adjust(left=.05, right=.95, bottom=.05, top=.9, 
        hspace=.2)

    # Suptitle
    if example_frames_to_plot is example_best_frames:
        figure_handle.suptitle('best frames')
    else:
        j_label, c_label = error_types_to_plot[n_error_type]
        translated_j_label = c_labels[list(j_labels).index(j_label)]
        figure_handle.suptitle('error frames: %d (%s, %s) labeled as %d (%s, %s)' % (
            c_label, curated_num2name[c_label], curated_colors[c_label],
            j_label, curated_num2name[translated_j_label], 
            curated_colors[translated_j_label]))

    axa_curated = axa[0]
    axa_test = axa[1]
    for ax_curated, ax_test, frame in zip(
        axa_curated, axa_test, example_frames_to_plot):
        ax_curated.set_title(str(frame), size='small')
        ax_test.set_title(str(frame), size='small')

    with tables.open_file(video_session.data.whiskers.get_path) as wfh:
        # Plot the curated
        whiskvid.output_video.plot_stills_with_overlays_from_data(
            monitor_video_filename=video_session.data.monitor_video.get_path,
            frame_triggers=example_frames_to_plot,
            whiskers_table=cwe_cdata,
            whiskers_file_handle=wfh,
            contacts_table=contacts_table,
            contact_colors=curated_colors,
            axa=axa_curated,
            )

    with tables.open_file(video_session.data.whiskers.get_path) as wfh:
        # Plot the test data
        whiskvid.output_video.plot_stills_with_overlays_from_data(
            monitor_video_filename=video_session.data.monitor_video.get_path,
            frame_triggers=example_frames_to_plot,
            whiskers_table=cwe_jdata, 
            whiskers_file_handle=wfh,
            contacts_table=contacts_table,
            contact_colors=test_set_colors,
            axa=axa_test,
            )    
plt.show()

