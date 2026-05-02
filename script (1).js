
async function run() {
    // Define a model for linear regression.
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));

    // Prepare the model for training: Specify the loss and the optimizer.
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    // Generate some synthetic data for training.
    const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]); // Input (x values)
    const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]); // Output (y values, assuming y = 2x - 1 roughly)

    // Train the model using the data.
    await model.fit(xs, ys, {epochs: 500});

    // Use the model to predict a value.
    const prediction = model.predict(tf.tensor2d([5], [1, 1]));

    const outputElement = document.getElementById('output');
    outputElement.innerText = `Prediction for x = 5: ${prediction.dataSync()[0]}`;

    console.log(`Prediction for x = 5: ${prediction.dataSync()[0]}`);
}

run();
