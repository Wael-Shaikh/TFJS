const tf = require("@tensorflow/tfjs-node");

const model = tf.sequential();

model.add(tf.layers.dense({units:8, inputShape: 2, activation: 'tanh'}));
model.add(tf.layers.dense({units:2, activation: 'softmax'}));

model.compile({
    loss: 'binaryCrossentropy',
    optimizer: 'adam',
    lr: 0.001,
    metrics: ['accuracy']
});

