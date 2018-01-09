import tensorflow as tf
from tensorforce.core.networks import Network


class Structure2Vector(Network):
    EMBEDDING_SIZE = 64
    EMBEDDING_ITERATIONS = 4
    OUTPUT_SIZE = 20

    def embed_state(self, vertex_states, ub_vertices):
        with tf.variable_scope("graph_embedding", reuse=True):
            theta_1_neg_inc = tf.get_variable(name="theta_1_neg_inc")
            theta_1_pos_inc = tf.get_variable(name="theta_1_pos_inc")

        # [ES, 1] x [1, NC] = [ES, NC]
        neg_inc_states = tf.reshape(tf.where(vertex_states == -1,
                                             tf.ones_like(vertex_states),
                                             tf.zeros_like(vertex_states)),
                                    [1, -1], name="neg_inc_states_reshaped")
        neg_inc_embedding = tf.matmul(theta_1_neg_inc, neg_inc_states, name="neg_inc_embedding")

        pos_inc_states = tf.reshape(tf.where(vertex_states == 1,
                                             tf.ones_like(vertex_states),
                                             tf.zeros_like(vertex_states)),
                                    [1, -1], name="pos_inc_states")
        pos_inc_embedding = tf.matmul(theta_1_pos_inc, pos_inc_states, name="pos_inc_embedding")

        state_embedding = tf.divide(tf.add(neg_inc_embedding, pos_inc_embedding), [2.0], name="state_embedding_average")
        state_embedding = tf.reshape(state_embedding, [self.EMBEDDING_SIZE, -1, ub_vertices], name="state_embedding_reshaped")
        state_embedding = tf.reduce_mean(state_embedding, axis=2, keep_dims=False, name="state_embedding_reduced_mean")

        return state_embedding

    def pool_neighbourhood(self, embedding, neighbourhoods, theta):
        pooled_neighbourhood = tf.matmul(embedding, neighbourhoods, name="pooled_neighbourhood")
        pooled_neighbourhood = tf.reshape(pooled_neighbourhood, [self.EMBEDDING_SIZE, -1],
                                          name="pooled_neighbourhood_reshaped")

        # [ES, ES] x [ES, NC] = [ES, NC]
        neighbourhood_embedding = tf.matmul(theta, pooled_neighbourhood, name="neighbourhood_embedding")
        return neighbourhood_embedding

    def embed_neighbourhoods(self, embedding, neighbourhoods):
        with tf.variable_scope("graph_embedding", reuse=True):
            theta_2 = tf.get_variable(name="theta_2")
        return self.pool_neighbourhood(embedding, neighbourhoods, theta_2)

    def pool_distances(self, distances, theta):
        distances = tf.reshape(distances, [1, -1], name="distances_reshaped")
        pooled_distances = tf.matmul(theta, distances, name="pooled_distances")
        return pooled_distances

    def embed_distances(self, distances, batch_size, ub_vertices):
        with tf.variable_scope("graph_embedding", reuse=True):
            theta_3 = tf.get_variable(name="theta_3")
            theta_4 = tf.get_variable(name="theta_4")

        pooled_distances = self.pool_distances(distances, theta_4)
        relu_pooled_distances = tf.nn.relu(pooled_distances, name="relu_pooled_distances")
        relu_pooled_distances = tf.reshape(relu_pooled_distances, [self.EMBEDDING_SIZE, batch_size * ub_vertices, ub_vertices],
                                           name="relu_pooled_distances_reshape")
        sum_relu_distances = tf.reduce_sum(relu_pooled_distances, -1, name="sum_relu_distances")

        # [ES, ES] x [ES, NC] = [ES, NC]
        distance_embedding = tf.matmul(theta_3, sum_relu_distances, name="distance_embedding")
        return distance_embedding

    def graph_embedding_layer(self, embedding, vertex_states, neighbourhoods, distances, batch_size, ub_vertices):
        embedded_state = self.embed_state(vertex_states, ub_vertices)
        embedded_neighbourhoods = self.embed_neighbourhoods(embedding, neighbourhoods)
        embedded_distances = self.embed_distances(distances, batch_size, ub_vertices)

        # [ES, NC] + [ES, NC] + [ES, NC] = [ES, NC]
        new_embedding = tf.add_n([embedded_state,
                                  embedded_neighbourhoods,
                                  embedded_distances], name="new_embedding_sum_pool")
        return tf.reshape(new_embedding, [batch_size, self.EMBEDDING_SIZE, ub_vertices], name="new_embedding_reshape")

    def embedding2Qfunction(self, embedding, neighbourhoods):
        with tf.variable_scope("q_function", reuse=True):
            theta_5 = tf.get_variable(name="theta_5")
            theta_6 = tf.get_variable(name="theta_6")
            theta_7 = tf.get_variable(name="theta_7")

        # [ES, ES] x [ES, NC] = [ES, NC]
        embedded_neighbourhoods = self.pool_neighbourhood(embedding, neighbourhoods, theta_6)

        # [ES, ES] x [ES, NC] = [ES, NC]
        embedding = tf.reshape(embedding, [self.EMBEDDING_SIZE, -1], name="embedding_reshape")
        theta_7_x_embedding = tf.matmul(theta_7, embedding, name="theta_7_x_embedding")

        # [[ES, NC], [ES, NC]] = [2ES, NC]
        concat_theta_6_theta_7 = tf.concat([embedded_neighbourhoods, theta_7_x_embedding], axis=0,
                                           name="concat_theta_6_theta_7")
        relu_concat = tf.nn.relu(concat_theta_6_theta_7, name="relu_concat")

        # [1, 2ES] x [2ES, NC] = [1, NC]
        q_hat = tf.matmul(theta_5, relu_concat, transpose_a=True, name="q_hat")
        # q_hat = tf.multiply(q_hat, [-1], name="negative_q_hat")
        return q_hat

    def tf_apply(self, x, internals=(), update=False, return_internals=False):
        var_states = x["vertex_states"]
        neighbourhoods = x["distances"]
        distances = x["neighbourhoods"]

        batch_size = tf.shape(var_states)[0]

        embedding = tf.zeros((batch_size, self.EMBEDDING_SIZE, self.OUTPUT_SIZE), dtype=tf.float32)

        initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)

        with tf.variable_scope("graph_embedding"):
            theta_1_neg_inc = tf.get_variable(name="theta_1_neg_inc",
                                              shape=(self.EMBEDDING_SIZE, 1),
                                              initializer=initializer)
            theta_1_pos_inc = tf.get_variable(name="theta_1_pos_inc",
                                              shape=(self.EMBEDDING_SIZE, 1),
                                              initializer=initializer)
            theta_2 = tf.get_variable(name="theta_2",
                                      shape=(self.EMBEDDING_SIZE, self.EMBEDDING_SIZE),
                                      initializer=initializer)
            theta_3 = tf.get_variable(name="theta_3",
                                      shape=(self.EMBEDDING_SIZE, self.EMBEDDING_SIZE),
                                      initializer=initializer)
            theta_4 = tf.get_variable(name="theta_4",
                                      shape=(self.EMBEDDING_SIZE, 1),
                                      initializer=initializer)

        with tf.variable_scope("q_function"):
            theta_5 = tf.get_variable(name="theta_5",
                                      shape=(2 * self.EMBEDDING_SIZE, 1),
                                      initializer=initializer)
            theta_6 = tf.get_variable(name="theta_6",
                                      shape=(self.EMBEDDING_SIZE, self.EMBEDDING_SIZE),
                                      initializer=initializer)
            theta_7 = tf.get_variable(name="theta_7",
                                      shape=(self.EMBEDDING_SIZE, self.EMBEDDING_SIZE),
                                      initializer=initializer)

        for it in range(self.EMBEDDING_ITERATIONS):
            embedding = self.graph_embedding_layer(embedding, var_states, neighbourhoods, distances, batch_size, self.OUTPUT_SIZE)
            embedding = tf.nn.relu(embedding, name="relu_embedding_" + str(it))

        q_hat = self.embedding2Qfunction(embedding, neighbourhoods)
        q_hat = tf.reshape(q_hat, [batch_size, self.OUTPUT_SIZE], name="q_hat_reshaped")

        if return_internals:
            return q_hat, []
        else:
            return q_hat
