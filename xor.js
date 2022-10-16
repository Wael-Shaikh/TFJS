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

function onBatchEnd(batch, logs) {
  //console.log("Loss:", logs.loss);
  //console.log("Accuracy:", logs.acc);
}

model.fit(x, y, {epochs: 1000, callbacks: {onBatchEnd}}).then((info)  => {
  console.log("Model Summary:");
  model.summary();
  //console.log("Accuracy:", info);
  model.predict(tf.tensor2d([[0,0]])).argMax(axis=1).print();
  model.predict(tf.tensor2d([[0,1]])).argMax(axis=1).print();
  model.predict(tf.tensor2d([[1,0]])).argMax(axis=1).print();
  model.predict(tf.tensor2d([[1,1]])).argMax(axis=1).print();
});

