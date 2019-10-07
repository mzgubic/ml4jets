import numpy as np
import pandas as pd
import pyjet as pj


max_n_jets = 10
n_hadrons = 700
jet_ptmin = 20
dataset_path = '/data/atlassmallfiles/users/zgubic/OT4NP/events_anomalydetection.h5'


def hadron_generator(path, batch_size, total_size=1100000, sort=False):
    """
    Yields batches of hadron level events.

    Arguments:
        path (string): path to the raw file
        batch_size (int): number of events per batch
        total_size (int): total number of events in the file
        sort (string): sort hadrons accoridng to pt: [False, 'ascending', 'descending']

    Returns:
        x (float array): shape (batch_size, n_hadrons, 3) array of hadron properties (pt, eta, phi)
        y (float array): shape (batch_size, 1) signal (1) or background (0)
    """
    
    i = 0
    while i+batch_size <= total_size:

        # read
        batch = pd.read_hdf(dataset_path, start=i, stop=i+batch_size).to_numpy()
        i+=batch_size

        # extract
        x = batch[:, :-1].reshape(-1, n_hadrons, 3)
        y = batch[:, -1].reshape(-1, 1)

        # get indices for sorting
        if sort == 'ascending':
            indices_1 = np.argsort(x[:, :, 0])
        if sort == 'descending':
            indices_1 = np.argsort(x[:, :, 0])[:, ::-1]

        # do the sorting
        if sort:
            indices_0 = np.repeat(np.arange(batch_size), n_hadrons)
            x = x[indices_0, indices_1.reshape(-1), :].reshape(-1, n_hadrons, 3)

        yield x, y


def cluster_jets(event, R, p):
    """
    Cluster jets from hadron level event.

    Arguments:
        event (int array): 
        R (float): jet cluster radius
        p (float): jet cluster algo parameter
    """

    # create the clustering inputs
    event = event.reshape(-1, 3)
    nhad = sum(event[:,0] > 0)
    pjets = np.zeros(nhad, dtype=pj.DTYPE_PTEPM)
    for had in range(nhad):
        pjets[had]['pT'] = event[had, 0]
        pjets[had]['eta'] = event[had, 1]
        pjets[had]['phi'] = event[had, 2]

    # cluster to jets
    sequence = pj.cluster(pjets, R=R, p=p)
    jets = sequence.inclusive_jets(ptmin=jet_ptmin)

    # return the jets array
    jet_array = np.zeros(shape=(len(jets), 3))
    for i, jet in enumerate(jets):
        jet_array[i, 0] = jet.pt
        jet_array[i, 1] = jet.eta
        jet_array[i, 2] = jet.phi

    return jet_array


def jet_generator(path, batch_size, total_size=1100000, ascending=True):
    """
    Yields batches of jet level events.

    Arguments:
        path (string): path to the raw file
        batch_size (int): number of events per batch
        total_size (int): total number of events in the file
        ascending (bool): sort jets in ascending pt

    Returns:
        x (float array): shape (batch_size, n_jets, 3) array of hadron properties (pt, eta, phi)
        y (float array): shape (batch_size, 1) signal (1) or background (0)
    """

    for hadron_events, truth in hadron_generator(path, batch_size, total_size):

        # cluster the hadrons
        jet_events = np.zeros(shape=(batch_size, max_n_jets, 3))
        for i, event in enumerate(hadron_events):

            # get variable length sequence
            jets = cluster_jets(event, R=0.4, p=-1)

            # cut down to max number
            jets = jets[:max_n_jets, :]

            # and fill with zeros
            jets = np.pad(jets, pad_width=((0, max_n_jets-len(jets)), (0, 0)) )

            # sort according to pt
            if ascending:
                jets = np.flip(jets, axis=0)

            # append to the array
            jet_events[i] = jets

        yield jet_events, truth


batch_size=2
hadgen = hadron_generator(dataset_path, batch_size=batch_size, total_size=30, sort='ascending')
#hadgen = hadron_generator(dataset_path, batch_size=batch_size, total_size=30, sort='descending')
#hadgen = hadron_generator(dataset_path, batch_size=batch_size, total_size=30, sort=False)
jetgen = jet_generator(dataset_path, batch_size=batch_size, total_size=30, ascending=False)

#for x, y in hadgen:
for x, y in jetgen:
    print(y)
    print(x)
    break
    pass
    
    
    
    



