// Random Forest implementation

// Random Forest model state
let randomForestModel = null;

// Initialize when document is loaded
document.addEventListener('DOMContentLoaded', function () {
    // Set up event listeners
    document.getElementById('build-forest-btn').addEventListener('click', buildRandomForest);

    // Update number of trees display on slider change
    document.getElementById('n-trees').addEventListener('input', function () {
        document.getElementById('n-trees-value').textContent = this.value;
    });
});

// Build a random forest based on current dataset and parameters
function buildRandomForest() {
    const numTrees = parseInt(document.getElementById('n-trees').value);
    const maxFeaturesType = document.getElementById('max-features').value;
    const useBootstrap = document.getElementById('bootstrap').checked;

    if (currentDataset.length === 0) {
        alert('Please select a dataset first');
        return;
    }

    // Split data into training and testing sets
    const { trainData, trainLabels, testData, testLabels } = splitData(currentDataset, currentLabels);

    // Start timer
    const forestStart = performance.now();

    // Create forest
    randomForestModel = {
        trees: [],
        featureImportances: Array(trainData[0].length).fill(0)
    };

    // Build each tree
    for (let i = 0; i < numTrees; i++) {
        // Prepare bootstrap sample if enabled
        let treeData = trainData;
        let treeLabels = trainLabels;

        if (useBootstrap) {
            const bootstrapSample = createBootstrapSample(trainData, trainLabels);
            treeData = bootstrapSample.data;
            treeLabels = bootstrapSample.labels;
        }

        // Determine max features for this tree
        const numFeatures = treeData[0].length;
        let maxFeatures;

        switch (maxFeaturesType) {
            case 'sqrt':
                maxFeatures = Math.max(1, Math.floor(Math.sqrt(numFeatures)));
                break;
            case 'log2':
                maxFeatures = Math.max(1, Math.floor(Math.log2(numFeatures)));
                break;
            default:
                maxFeatures = numFeatures;
        }

        // Sample random feature subset if needed
        let featureIndices = Array.from(Array(numFeatures).keys());
        if (maxFeatures < numFeatures) {
            shuffleArray(featureIndices);
            featureIndices = featureIndices.slice(0, maxFeatures);
        }

        // Create a version of the data with only the selected features
        const featureSubsetData = treeData.map(sample =>
            featureIndices.map(idx => sample[idx])
        );

        // Build the tree with the feature subset
        const tree = buildTreeRecursive(
            featureSubsetData,
            treeLabels,
            0,
            parseInt(document.getElementById('max-depth').value),
            parseInt(document.getElementById('min-samples-split').value),
            document.getElementById('impurity-measure').value,
            featureIndices.map(idx => featureNames[idx])
        );

        // Store the feature mapping for prediction
        tree.featureIndices = featureIndices;

        // Add tree to the forest
        randomForestModel.trees.push(tree);

        // Update progress
        const progress = (i + 1) / numTrees * 100;
        updateForestProgress(progress);
    }

    const forestEnd = performance.now();

    // Calculate feature importances across all trees
    calculateFeatureImportance(randomForestModel);

    // Evaluate the forest
    const treeAccuracy = evaluateTree(
        randomForestModel.trees[0],
        testData.map(sample => randomForestModel.trees[0].featureIndices.map(idx => sample[idx])),
        testLabels
    );

    const forestAccuracy = evaluateForest(randomForestModel, testData, testLabels);

    // Update forest stats
    document.getElementById('forest-stats').innerHTML = `
        <p><strong>Random Forest Built Successfully</strong></p>
        <p>Number of Trees: ${numTrees}</p>
        <p>Max Features: ${maxFeaturesType}</p>
        <p>Bootstrap Sampling: ${useBootstrap ? 'Yes' : 'No'}</p>
        <p>Training Time: ${(forestEnd - forestStart).toFixed(2)} ms</p>
        <p>Single Tree Accuracy: ${(treeAccuracy * 100).toFixed(2)}%</p>
        <p>Forest Accuracy: ${(forestAccuracy * 100).toFixed(2)}%</p>
    `;

    // Draw forest decision boundary
    drawForestBoundary(randomForestModel, currentDataset, currentLabels);

    // Visualize a sample of trees from the forest
    visualizeForestSample(randomForestModel);

    // Draw feature importance chart
    drawFeatureImportanceChart(featureNames, randomForestModel.featureImportances);

    // Draw performance comparison chart
    drawPerformanceComparison(treeAccuracy, forestAccuracy);

    return randomForestModel;
}

// Create a bootstrap sample from the data
function createBootstrapSample(data, labels) {
    const bootstrapData = [];
    const bootstrapLabels = [];

    for (let i = 0; i < data.length; i++) {
        const idx = Math.floor(Math.random() * data.length);
        bootstrapData.push(data[idx]);
        bootstrapLabels.push(labels[idx]);
    }

    return { data: bootstrapData, labels: bootstrapLabels };
}

// Shuffle array in-place
function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

// Update forest building progress
function updateForestProgress(percentage) {
    // You can implement a progress bar here if needed
    console.log(`Forest building progress: ${percentage.toFixed(0)}%`);
}

// Calculate feature importance across all trees
function calculateFeatureImportance(forest) {
    // Reset importances
    forest.featureImportances = Array(featureNames.length).fill(0);

    // Sum up feature importances across all trees
    for (const tree of forest.trees) {
        calculateNodeImportance(tree, forest.featureImportances);
    }

    // Normalize importances
    const totalImportance = forest.featureImportances.reduce((sum, imp) => sum + imp, 0);
    if (totalImportance > 0) {
        forest.featureImportances = forest.featureImportances.map(imp => imp / totalImportance);
    }
}

// Recursively calculate importance for a tree
function calculateNodeImportance(node, importances) {
    if (node.isLeaf) return;

    // Add this node's importance (gain * samples) to the global array
    const nodeImportance = node.gain * node.samples;

    // Map from the tree's feature subset to the original feature indices
    const originalFeatureIndex = node.featureIndices ?
        node.featureIndices[node.featureIndex] : node.featureIndex;

    importances[originalFeatureIndex] += nodeImportance;

    // Recurse on children
    calculateNodeImportance(node.left, importances);
    calculateNodeImportance(node.right, importances);
}

// Predict class with random forest
function predictWithForest(forest, sample) {
    if (!forest || !forest.trees || forest.trees.length === 0) return null;

    // Collect votes from each tree
    const votes = {};

    for (const tree of forest.trees) {
        // Map sample to the tree's feature subset
        const treeFeatures = tree.featureIndices.map(idx => sample[idx]);

        const prediction = predictWithTree(tree, treeFeatures);
        votes[prediction] = (votes[prediction] || 0) + 1;
    }

    // Find the class with the most votes
    let maxVotes = -1;
    let majorityClass = null;

    for (const [cls, count] of Object.entries(votes)) {
        if (count > maxVotes) {
            maxVotes = count;
            majorityClass = parseInt(cls);
        }
    }

    return majorityClass;
}

// Evaluate forest on test data
function evaluateForest(forest, testData, testLabels) {
    let correct = 0;

    for (let i = 0; i < testData.length; i++) {
        const prediction = predictWithForest(forest, testData[i]);
        if (prediction === testLabels[i]) {
            correct++;
        }
    }

    return correct / testData.length;
}
