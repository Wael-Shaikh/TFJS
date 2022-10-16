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

x = [[0,0], [0,1], [1,0], [1,1]];
y = [[0], [1], [1], [0]];
y = [[1,0],[0,1],[0,1],[1,0]];

y = tf.tensor2d(y);
x = tf.tensor2d(x);

//console.log(x);
//console.log(y);

