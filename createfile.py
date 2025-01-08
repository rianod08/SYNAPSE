import os
import numpy as np
!pip install mne
import mne

def extract_eeg_samples_mne(data_dir, output_file="/content/drive/MyDrive/eeg_dataset.npz"):

    all_samples = []
    all_labels = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".bdf"):
            filepath = os.path.join(data_dir, filename)

            try:
                raw = mne.io.read_raw_bdf(filepath, preload=True)
                print(f"Processing file: {filename}")
                print(f"Available channels: {raw.ch_names}")

               
                if "Status" in raw.ch_names:
                    stim_channel = "Status"
                else:
                    print(f"No 'Status' channel found in {filename}. Skipping file.")
                    continue

              
                events = mne.find_events(raw, stim_channel=stim_channel)
                print(f"Found {len(events)} events in {filename}")
                if filename == 'R_S3_B5.bdf':
                    events = events[20:]  
                for event in events:
                    onset_sample, _, trigger = event
                    print(f"Processing trigger: {trigger}")
                   


                    
                    sfreq = int(raw.info["sfreq"]) 
                    sample_start = onset_sample
                    sample_end = sample_start + int(2 * sfreq)

                    if sample_end <= raw.n_times:
                        sample = raw.get_data(start=sample_start, stop=sample_end)
                        all_samples.append(sample)
                        all_labels.append(trigger)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Convert to numpy arrays
    all_samples = np.array(all_samples)
    all_labels = np.array(all_labels)

    for i in range(1, len(all_labels)):
        if all_labels[i] == 10 and all_labels[i - 1] == 10:
            all_labels[i] = 0

    def exclude_tens(labels, samples):

        labels = np.array(labels)
        samples = np.array(samples)

        if len(labels) != len(samples):
            raise ValueError("Labels and samples must have the same length.")

        filtered_labels = []
        filtered_samples = []

        for label, sample in zip(labels, samples):
            if label != 10 and label != 65536:
                filtered_labels.append(label)
                filtered_samples.append(sample)

        return filtered_labels, filtered_samples

    labels, samples = exclude_tens(all_labels, all_samples)
    # Save dataset
    np.savez(output_file, samples=samples, labels=labels)
    print(f"Dataset saved to {output_file}")

def load_eeg_dataset(npz_file):

    data = np.load(npz_file)
    samples = data['samples']
    labels = data['labels']
    return samples, labels


data_directory = "/content/drive/MyDrive/BDF data/EEG PC/Session_2"
extract_eeg_samples_mne(data_directory)


dataset_file = "eeg_dataset.npz"
samples, labels = load_eeg_dataset(dataset_file)
print(samples.shape, labels.shape)
print(labels)
print(f"Loaded {samples.shape[0]} samples with shape {samples.shape[1:]} and {len(labels)} labels.")



