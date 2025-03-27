// Main application file for perceptron demo

// Wait for DOM content to load
document.addEventListener("DOMContentLoaded", function () {
    // Initialize visualizations
    initCanvases();

    // Create model
    createModel();

    // Draw initial diagrams
    drawPerceptronDiagram();
    updateWeightsVisualization();

    // Generate initial data
    const { trainingData, trainingLabels, slope, intercept } = generateData();
    drawData(trainingData, trainingLabels, slope, intercept);

    // Set up event listeners
    document.getElementById("generate-btn").addEventListener("click", () => {
        const { trainingData, trainingLabels, slope, intercept } = generateData();
        createModel();
        document.getElementById("accuracy-info").textContent = "";
        drawData(trainingData, trainingLabels, slope, intercept);
    });

    document.getElementById("train-btn").addEventListener("click", () => {
        trainModel();
    });

    document.getElementById("step-btn").addEventListener("click", () => {
        trainStepByStep();
    });

    document.getElementById("reset-btn").addEventListener("click", () => {
        const { model, currentEpoch } = resetModelState();
        const { trainingData, trainingLabels, slope, intercept } = getModelState();
        drawData(trainingData, trainingLabels, slope, intercept);
    });

    // Weight visualization controls
    document
        .getElementById("weight1")
        .addEventListener("input", updateWeightsVisualization);
    document
        .getElementById("weight2")
        .addEventListener("input", updateWeightsVisualization);
    document
        .getElementById("bias")
        .addEventListener("input", updateWeightsVisualization);
});
