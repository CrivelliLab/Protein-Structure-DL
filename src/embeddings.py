import tensorflow as tf
from datasets import get_datasets, DataLoader
from trainers.classifier_trainer import ClassifierTrainer
import yaml
from os.path import join as path_join
import numpy as np
import os
from  scipy.spatial.distance import euclidean, cosine
from scipy.stats import percentileofscore as perc


residues = ['ALA', 'ARG', 'ASN', 'ASP', 'ASX', 'CYS', 'GLN',
            'GLU', 'GLX', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
            'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR',
            'UNK', 'VAL']


class EmbeddingExtractor(ClassifierTrainer):
    def __init__(self, checkpoint_path, **kwargs):
        self.checkpoint_path = checkpoint_path
        super().__init__(**kwargs)

    def get_embedding(self, data_loader: DataLoader):
        # Start the session
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # Setup the variable initialisation
            sess.run(tf.global_variables_initializer())

            # load pretrained model
            saver.restore(sess, self.checkpoint_path)

            # Loop over batches
            self.logger.info('Getting protein embeddings...')
            embeddings = np.array([])
            for i, data in enumerate(data_loader):
                data = [False, ] + data
                out = sess.run(self.operators[2:4], feed_dict={i: d for i, d in zip(self.inputs, data)})
                if embeddings.shape[0] == 0:
                    embeddings = out[1]
                else:
                    embeddings = np.vstack((embeddings, out[1]))

        return embeddings


def rgn_output_to_gcn_input(data_path: str):
    rgn_data_path = path_join(data_path, 'rgn_output')
    for filename in os.listdir(rgn_data_path):
        coords = np.loadtxt(path_join(rgn_data_path, filename))
        coords = coords.reshape(-1, 3, 3)
        coords = coords[:, 1, :].squeeze()
        seq_len = coords.shape[0]
        out_data = np.zeros((seq_len, 10))
        centroid = np.mean(coords, axis=0)
        residue_depths = np.array([euclidean(centroid, c) for c in coords])
        for i, c in enumerate(coords):
            # residue coordinates
            out_data[i][-3:] = c
            # mask (all ones)
            out_data[i][0] = 1
            # amino acid number in protein sequence
            out_data[i][1] = i+1
            # amino acid encoding number
            out_data[i][2] = 6
            # residue depth percentile
            out_data[i][3] = 1 - perc(residue_depths, residue_depths[i]) / 100.0
            # residue orientation (we currently don't have side chain info so this ends up being 0)
            sc_c = c
            c = c - centroid
            sc_c = sc_c - centroid
            sc_c = sc_c - c
            c = -c
            out_data[i][4] = 1 - cosine(c, sc_c)
            if np.isnan(out_data[i][4]): out_data[i][4] = 0
        out_data = np.concatenate([out_data[:, :3].astype(int).astype(str),
                                   out_data[:, 3:5],
                                   out_data[:, 5:7].astype(int).astype(str),
                                   out_data[:, -3:]], axis=1)

        # Save graph
        new_filename = '.'.join([filename.split('.')[0], 'txt'])
        graph_data_path = path_join(data_path, 'graph')
        if not os.path.exists(graph_data_path): os.mkdir(graph_data_path)
        with open(path_join(data_path, 'graph', new_filename), 'w') as f:
            for i, _ in enumerate(out_data):
                f.write(' '.join(_) + '\n')


if __name__ == "__main__":

    rgn_output_to_gcn_input('./data/test/')
    assert 1 == 2

    path_to_data = './data/Oncogenes/graph/'

    with open('./src/configs/oncogenes/oncogenes_graph_embed.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    data_config = config['data_config']
    data_config['data_path'] = path_to_data
    model_config = config.get('model_config', {})
    train_config = config['train_config']
    experiment_config = config['experiment_config']
    output_dir = experiment_config.pop('output_dir', None)

    dataset = get_datasets(**data_config, evaluate=True)

    batch_size = train_config.pop('batch_size')

    data_loader = DataLoader(dataset, batch_size=batch_size, cores=1)

    name = experiment_config.pop('name')
    emb_ex = EmbeddingExtractor(checkpoint_path=path_join(output_dir, 'model/model.ckpt'),
                                output_dir=output_dir, **experiment_config)
    emb_ex.build_model(**model_config)
    emb_ex.print_model_summary()
    embeddings = emb_ex.get_embedding(data_loader)
    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    print(embeddings.shape)
    print(embeddings[0, 0:10])
    print(embeddings[1, 0:10])


