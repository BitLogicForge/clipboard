// Model module for perceptron demo

// Model parameters
let model;
let trainingData = [];
let trainingLabels = [];
let currentEpoch = 0;
let totalEpochs = 50;
let trainingInProgress = false;
let stepByStepMode = false;

// Line parameters for visualization
let slope = 0;
let intercept = 0;

// Create a simple perceptron model
function createModel() {
    model = tf.sequential();
    model.add(
        tf.layers.dense({
            units: 1,
            inputShape: [2],
            activation: "sigmoid",
        })
    );

    model.compile({
        optimizer: tf.train.sgd(0.1),
        loss: "binaryCrossentropy",
        metrics: ["accuracy"],
    });

    document.getElementById("model-info").textContent =
        "Model created but not trained";

    currentEpoch = 0;
    updateProgressBar(0);

    return model;
}

// Generate random data points
function generateData(count = 100) {
    trainingData = [];
    trainingLabels = [];

    const canvas = document.getElementById("plot-canvas");
    const width = canvas.width;
    const height = canvas.height;

    // Create a random line for classification
    const x1 = Math.random() * width;
    const y1 = Math.random() * height;
    const x2 = Math.random() * width;
    const y2 = Math.random() * height;

    // Calculate line parameters (y = mx + b)
    slope = (y2 - y1) / (x2 - x1);
    intercept = y1 - slope * x1;

    for (let i = 0; i < count; i++) {
        const x = Math.random() * width;
        const y = Math.random() * height;

        trainingData.push([x / width, y / height]); // Normalize data

        // Determine label based on which side of the line the point is on
        const expected = y < slope * x + intercept ? 0 : 1;
        trainingLabels.push(expected);
    }

    return { trainingData, trainingLabels, slope, intercept };
}

// Train the model step by step
async function trainStepByStep() {
    if (trainingData.length === 0) {
        alert("Generate data first!");
        return;
    }

    if (trainingInProgress) return;

    stepByStepMode = true;
    trainingInProgress = true;

    const xs = tf.tensor2d(trainingData);
    const ys = tf.tensor1d(trainingLabels);

    if (currentEpoch < totalEpochs) {
        document.getElementById("model-info").textContent = `Training step ${currentEpoch + 1
            }/${totalEpochs}...`;

        await model.trainOnBatch(xs, ys).then((logs) => {
            document.getElementById(
                "model-info"
            ).textContent = `Completed step ${currentEpoch + 1
                }/${totalEpochs} - Loss: ${logs.loss.toFixed(4)}`;

            currentEpoch++;
            updateProgressBar((currentEpoch / totalEpochs) * 100);

            // Visualize current model state
            visualizeModel(model, trainingData, trainingLabels, slope, intercept);
        });

        if (currentEpoch === totalEpochs) {
            const result = model.evaluate(xs, ys);
            const accuracy = result[1].dataSync()[0];
            document.getElementById(
                "accuracy-info"
            ).textContent = `Final accuracy: ${(accuracy * 100).toFixed(2)}%`;

            document.getElementById("model-info").textContent =
                "Training complete!";
            trainingInProgress = false;
        }
    } else {
        document.getElementById("model-info").textContent =
            "Training already complete!";
        trainingInProgress = false;
    }

    xs.dispose();
    ys.dispose();

    return currentEpoch;
}

// Train the model
async function trainModel() {
    if (trainingData.length === 0) {
        alert("Generate data first!");
        return;
    }

    if (trainingInProgress && !stepByStepMode) return;

    trainingInProgress = true;
    stepByStepMode = false;
    currentEpoch = 0;
    updateProgressBar(0);

    const xs = tf.tensor2d(trainingData);
    const ys = tf.tensor1d(trainingLabels);

    document.getElementById("model-info").textContent =
        "Training in progress...";

    await model.fit(xs, ys, {
        epochs: totalEpochs,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                currentEpoch = epoch + 1;
                updateProgressBar((currentEpoch / totalEpochs) * 100);

                document.getElementById(
                    "model-info"
                ).textContent = `Training... Epoch ${currentEpoch}/${totalEpochs} - Loss: ${logs.loss.toFixed(
                    4
                )} - Accuracy: ${logs.acc.toFixed(4)}`;
            },
        },
    });

    document.getElementById("model-info").textContent =
        "Training complete!";

    // Evaluate model
    const result = model.evaluate(xs, ys);
    const accuracy = result[1].dataSync()[0];
    document.getElementById(
        "accuracy-info"
    ).textContent = `Final accuracy: ${(accuracy * 100).toFixed(2)}%`;

    // Visualize decision boundary
    visualizeModel(model, trainingData, trainingLabels, slope, intercept);

    trainingInProgress = false;

    xs.dispose();
    ys.dispose();

    return { currentEpoch, accuracy };
}

// Reset the model state
function resetModelState() {
    createModel();
    currentEpoch = 0;
    trainingInProgress = false;
    stepByStepMode = false;
    document.getElementById("accuracy-info").textContent = "";
    return { model, currentEpoch };
}

// Get current model parameters
function getModelState() {
    return {
        model,
        trainingData,
        trainingLabels,
        currentEpoch,
        totalEpochs,
        trainingInProgress,
        stepByStepMode,
        slope,
        intercept
    };
}
