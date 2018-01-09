import tensorflow as tf

from tensorforce import util
from tensorforce.models import PGModel


class MCTSModel(PGModel):
    def tf_pg_loss_per_instance(self, states, internals, actions, terminal, reward, update):
        embedding = self.network.apply(x=states, internals=internals, update=update)
        log_probs = list()

        for name, distribution in self.distributions.items():
            logits, probabilities, state_value = distribution.parameterize(x=embedding)
            collapsed_size = util.prod(util.shape(logits)[1:])
            logits = tf.reshape(tensor=logits, shape=(-1, collapsed_size))
            log_probs.append(logits)
        logits = tf.reduce_mean(input_tensor=tf.concat(values=log_probs, axis=1), axis=1)
        return -logits * reward
