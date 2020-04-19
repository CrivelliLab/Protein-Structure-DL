import tensorflow as tf
from datasets import get_datasets, DataLoader
from trainers.classifier_trainer import ClassifierTrainer
import yaml
from os.path import join as path_join
import numpy as np


class EmbeddingExtractor(ClassifierTrainer):
    def __init__(self, checkpoint_path, **kwargs):
        self.checkpoint_path = checkpoint_path
        super().__init__(**kwargs)

    def get_embedding(self, data_loader):
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


if __name__ == "__main__":

    with open('./src/configs/oncogenes/oncogenes_graph_embed.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    data_config = config['data_config']
    #data_config['data_path'] = path_to_data
    #data_config['split'] = None
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


