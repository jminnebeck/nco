from core.baselines import Baseline
from core.networks import Network, Dense
import tensorflow as tf


class MCTSNetworkBaseline(Baseline):
    def __init__(self, network_spec, scope='network-baseline', summary_labels=()):
        """
        Network baseline.

        Args:
            network_spec: Network specification dict
        """
        self.network = Network.from_spec(
            spec=network_spec,
            kwargs=dict(summary_labels=summary_labels)
        )
        assert len(self.network.internal_inputs()) == 0

        self.nonlinear = Dense(size=1, activation="sigmoid", bias=0.0, scope='prediction')

        super(MCTSNetworkBaseline, self).__init__(scope, summary_labels)

    def tf_predict(self, states, update):
        embedding = self.network.apply(x=states, internals=(), update=update)
        prediction = self.nonlinear.apply(x=embedding)
        return tf.squeeze(input=prediction, axis=1)

    def tf_regularization_loss(self):
        """
        Creates the TensorFlow operations for the baseline regularization loss.

        Returns:
            Regularization loss tensor
        """
        regularization_loss = super(MCTSNetworkBaseline, self).tf_regularization_loss()
        if regularization_loss is None:
            losses = list()
        else:
            losses = [regularization_loss]

        regularization_loss = self.network.regularization_loss()
        if regularization_loss is not None:
            losses.append(regularization_loss)

        regularization_loss = self.nonlinear.regularization_loss()
        if regularization_loss is not None:
            losses.append(regularization_loss)

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None

    def get_variables(self, include_non_trainable=False):
        baseline_variables = super(MCTSNetworkBaseline, self).get_variables(include_non_trainable=include_non_trainable)
        network_variables = self.network.get_variables(include_non_trainable=include_non_trainable)
        layer_variables = self.nonlinear.get_variables(include_non_trainable=include_non_trainable)

        return baseline_variables + network_variables + layer_variables

    def get_summaries(self):
        baseline_summaries = super(MCTSNetworkBaseline, self).get_summaries()
        network_summaries = self.network.get_summaries()
        layer_summaries = self.nonlinear.get_summaries()

        return baseline_summaries + network_summaries + layer_summaries