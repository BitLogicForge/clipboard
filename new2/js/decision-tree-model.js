// Decision Tree model implementation

// Decision Tree model state
let decisionTreeModel = null;

// Initialize when document is loaded
document.addEventListener('DOMContentLoaded', function () {
    // Set up event listeners
    document.getElementById('build-tree-btn').addEventListener('click', buildDecisionTree);
    document.getElementById('predict-btn').addEventListener('click', predictSample);

    // Load initial dataset
    generateCircleDataset();
    visualizeDataset(currentDataset, currentLabels);
});

// Build a decision tree based on current dataset and parameters
function buildDecisionTree() {
    const maxDepth = parseInt(document.getElementById('max-depth').value);
    const minSamplesSplit = parseInt(document.getElementById('min-samples-split').value);
    const criterion = document.getElementById('impurity-measure').value;

    if (currentDataset.length === 0) {
        alert('Please select a dataset first');
        return;
    }

    // Split data into training and testing sets
    const { trainData, trainLabels, testData, testLabels } = splitData(currentDataset, currentLabels);

    // Build the tree
    const treeStart = performance.now();
    decisionTreeModel = buildTreeRecursive(
        trainData,
        trainLabels,
        0,
        maxDepth,
        minSamplesSplit,
        criterion,
        featureNames
    );
    const treeEnd = performance.now();

    // Evaluate the tree
    const accuracy = evaluateTree(decisionTreeModel, testData, testLabels);

    // Update tree info
    document.getElementById('tree-info').innerHTML = `
        <p><strong>Tree Built Successfully</strong></p>
        <p>Max Depth: ${maxDepth}</p>
        <p>Min Samples Split: ${minSamplesSplit}</p>
        <p>Criterion: ${criterion}</p>
        <p>Training Time: ${(treeEnd - treeStart).toFixed(2)} ms</p>
        <p>Test Accuracy: ${(accuracy * 100).toFixed(2)}%</p>
    `;

    // Visualize the tree
    visualizeTreeStructure(decisionTreeModel);

    // Draw decision boundary
    drawDecisionBoundary(decisionTreeModel, currentDataset, currentLabels);

    // Setup feature inputs for prediction
    setupFeatureInputs(featureNames);

    return decisionTreeModel;
}

// Build a decision tree recursively
function buildTreeRecursive(data, labels, depth, maxDepth, minSamplesSplit, criterion, featureNames) {
    // Count samples in each class
    const classCounts = {};
    for (const label of labels) {
        classCounts[label] = (classCounts[label] || 0) + 1;
    }

    // Find majority class
    let majorityClass = null;
    let maxCount = -1;
    for (const [cls, count] of Object.entries(classCounts)) {
        if (count > maxCount) {
            maxCount = count;
            majorityClass = parseInt(cls);
        }
    }

    // Create node with class distribution info
    const node = {
        samples: labels.length,
        classDistribution: classCounts,
        prediction: majorityClass
    };

    // Check if we should stop splitting
    if (depth >= maxDepth || labels.length < minSamplesSplit || Object.keys(classCounts).length === 1) {
        node.isLeaf = true;
        return node;
    }

    // Find the best split
    const bestSplit = findBestSplit(data, labels, criterion);

    // If no good split is found, make this a leaf node
    if (!bestSplit || bestSplit.gain <= 0) {
        node.isLeaf = true;
        return node;
    }

    // Add split info to the node
    node.feature = featureNames[bestSplit.featureIndex];
    node.featureIndex = bestSplit.featureIndex;
    node.threshold = bestSplit.threshold;
    node.gain = bestSplit.gain;
    node.impurity = bestSplit.impurity;
    node.criterion = criterion;
    node.leftSamples = bestSplit.leftLabels.length;
    node.rightSamples = bestSplit.rightLabels.length;

    // Split the data
    const leftData = [];
    const rightData = [];
    const leftLabels = [];
    const rightLabels = [];

    for (let i = 0; i < data.length; i++) {
        if (data[i][bestSplit.featureIndex] <= bestSplit.threshold) {
            leftData.push(data[i]);
            leftLabels.push(labels[i]);
        } else {
            rightData.push(data[i]);
            rightLabels.push(labels[i]);
        }
    }

    // Recursively build left and right subtrees
    node.left = buildTreeRecursive(leftData, leftLabels, depth + 1, maxDepth, minSamplesSplit, criterion, featureNames);
    node.right = buildTreeRecursive(rightData, rightLabels, depth + 1, maxDepth, minSamplesSplit, criterion, featureNames);

    return node;
}

// Find the best split for a dataset
function findBestSplit(data, labels, criterion) {
    if (data.length === 0) return null;

    const numFeatures = data[0].length;
    let bestGain = -Infinity;
    let bestFeatureIndex = null;
    let bestThreshold = null;
    let bestImpurity = null;
    let bestLeftLabels = null;
    let bestRightLabels = null;

    // Calculate current impurity
    const currentImpurity = criterion === 'gini' ?
        calculateGiniImpurity(labels) :
        calculateEntropy(labels);

    // Try each feature
    for (let featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
        // Get unique values for this feature
        const values = data.map(d => d[featureIndex]).sort((a, b) => a - b);
        const uniqueValues = [...new Set(values)];

        // For each unique value, try it as a threshold
        for (const threshold of uniqueValues) {
            // Split the data
            const leftLabels = [];
            const rightLabels = [];

            for (let i = 0; i < data.length; i++) {
                if (data[i][featureIndex] <= threshold) {
                    leftLabels.push(labels[i]);
                } else {
                    rightLabels.push(labels[i]);
                }
            }

            // Skip if either side is empty
            if (leftLabels.length === 0 || rightLabels.length === 0) continue;

            // Calculate impurity and information gain
            const leftImpurity = criterion === 'gini' ?
                calculateGiniImpurity(leftLabels) :
                calculateEntropy(leftLabels);

            const rightImpurity = criterion === 'gini' ?
                calculateGiniImpurity(rightLabels) :
                calculateEntropy(rightLabels);

            const leftWeight = leftLabels.length / labels.length;
            const rightWeight = rightLabels.length / labels.length;

            const weightedImpurity = leftWeight * leftImpurity + rightWeight * rightImpurity;
            const gain = currentImpurity - weightedImpurity;

            // Update best split if this one is better
            if (gain > bestGain) {
                bestGain = gain;
                bestFeatureIndex = featureIndex;
                bestThreshold = threshold;
                bestImpurity = currentImpurity;
                bestLeftLabels = leftLabels;
                bestRightLabels = rightLabels;
            }
        }
    }

    if (bestFeatureIndex === null) return null;

    return {
        featureIndex: bestFeatureIndex,
        threshold: bestThreshold,
        gain: bestGain,
        impurity: bestImpurity,
        leftLabels: bestLeftLabels,
        rightLabels: bestRightLabels
    };
}

// Calculate Gini impurity
function calculateGiniImpurity(labels) {
    const counts = {};
    for (const label of labels) {
        counts[label] = (counts[label] || 0) + 1;
    }

    let impurity = 1;
    for (const count of Object.values(counts)) {
        const p = count / labels.length;
        impurity -= p * p;
    }

    return impurity;
}

// Calculate entropy
function calculateEntropy(labels) {
    const counts = {};
    for (const label of labels) {
        counts[label] = (counts[label] || 0) + 1;
    }

    let entropy = 0;
    for (const count of Object.values(counts)) {
        const p = count / labels.length;
        entropy -= p * Math.log2(p);
    }

    return entropy;
}

// Predict class for a single sample
function predictWithTree(tree, sample) {
    let node = tree;

    // Traverse the tree until reaching a leaf node
    while (!node.isLeaf) {
        if (sample[node.featureIndex] <= node.threshold) {
            node = node.left;
        } else {
            node = node.right;
        }
    }

    return node.prediction;
}

// Evaluate tree on test data
function evaluateTree(tree, testData, testLabels) {
    let correct = 0;

    for (let i = 0; i < testData.length; i++) {
        const prediction = predictWithTree(tree, testData[i]);
        if (prediction === testLabels[i]) {
            correct++;
        }
    }

    return correct / testData.length;
}

// Get the decision path for a sample
function getDecisionPath(tree, sample) {
    const path = [];
    let node = tree;

    // Traverse the tree until reaching a leaf node
    while (true) {
        path.push(node);

        if (node.isLeaf) {
            break;
        }

        if (sample[node.featureIndex] <= node.threshold) {
            node = node.left;
        } else {
            node = node.right;
        }
    }

    return path;
}

// Predict a sample from user inputs
function predictSample() {
    if (!decisionTreeModel) {
        alert('Please build a decision tree first');
        return;
    }

    // Get feature values from inputs
    const sampleFeatures = [];
    for (let i = 0; i < featureNames.length; i++) {
        const value = parseFloat(document.getElementById(`feature-${i}`).value);
        if (isNaN(value)) {
            alert(`Please enter a valid number for ${featureNames[i]}`);
            return;
        }
        sampleFeatures.push(value);
    }

    // Get the decision path
    const path = getDecisionPath(decisionTreeModel, sampleFeatures);

    // Make prediction
    const prediction = predictWithTree(decisionTreeModel, sampleFeatures);

    // Visualize the path
    visualizePredictionPath(decisionTreeModel, sampleFeatures, path);

    return prediction;
}
