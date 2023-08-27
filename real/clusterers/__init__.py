from clusterers import (base_clusterer, selfcondgan, random_labels, online, dyngan)

clusterer_dict = {
    'supervised': base_clusterer.BaseClusterer,
    'selfcondgan': selfcondgan.Clusterer,
    'online': online.Clusterer,
    'random_labels': random_labels.Clusterer, 
    'dyngan': dyngan.Clusterer, 
}
