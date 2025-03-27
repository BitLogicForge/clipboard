// Data generation for decision tree demo

// Dataset state
let currentDataset = [];
let currentLabels = [];
let featureNames = ["x", "y"];

// Generate circle dataset
function generateCircleDataset(numPoints = 200) {
    const data = [];
    const labels = [];

    // Generate points in a circle pattern
    for (let i = 0; i < numPoints; i++) {
        // Random angle and radius
        const angle = Math.random() * Math.PI * 2;
        const radius = Math.random();

        // Convert to x,y coordinates
        const x = 0.5 + Math.cos(angle) * radius;
        const y = 0.5 + Math.sin(angle) * radius;

        // Label: inside small circle = 1, outside = 0
        const distFromCenter = Math.sqrt(Math.pow(x - 0.5, 2) + Math.pow(y - 0.5, 2));
        const label = distFromCenter < 0.3 ? 1 : 0;

        data.push([x, y]);
        labels.push(label);
    }

    currentDataset = data;
    currentLabels = labels;

    return { data, labels, featureNames };
}

// Generate XOR dataset
function generateXORDataset(numPoints = 200) {
    const data = [];
    const labels = [];

    // Generate random points
    for (let i = 0; i < numPoints; i++) {
        const x = Math.random();
        const y = Math.random();

        // XOR logic: if both x,y are on the same side of 0.5, label=0, else label=1
        const label = (x > 0.5 && y > 0.5) || (x < 0.5 && y < 0.5) ? 0 : 1;

        data.push([x, y]);
        labels.push(label);
    }

    currentDataset = data;
    currentLabels = labels;

    return { data, labels, featureNames };
}

// Generate spiral dataset
function generateSpiralDataset(numPoints = 200) {
    const data = [];
    const labels = [];
    const n = Math.floor(numPoints / 2);

    // Generate two interleaving spirals
    for (let i = 0; i < n; i++) {
        // First spiral
        const r1 = i / n * 0.4;
        const theta1 = i / n * 6 * Math.PI;

        const x1 = 0.5 + r1 * Math.cos(theta1);
        const y1 = 0.5 + r1 * Math.sin(theta1);

        data.push([x1, y1]);
        labels.push(0);

        // Second spiral
        const r2 = i / n * 0.4;
        const theta2 = i / n * 6 * Math.PI + Math.PI;

        const x2 = 0.5 + r2 * Math.cos(theta2);
        const y2 = 0.5 + r2 * Math.sin(theta2);

        data.push([x2, y2]);
        labels.push(1);
    }

    currentDataset = data;
    currentLabels = labels;

    return { data, labels, featureNames };
}

// Generate random dataset
function generateRandomDataset(numPoints = 200) {
    const data = [];
    const labels = [];

    // Generate a random linear boundary for classification
    const slope = Math.random() * 2 - 1; // Random slope between -1 and 1
    const intercept = Math.random(); // Random intercept between 0 and 1

    // Generate random points
    for (let i = 0; i < numPoints; i++) {
        const x = Math.random();
        const y = Math.random();

        // Determine label based on which side of line point is on
        // Line equation: y = slope * x + intercept
        const label = y > slope * x + intercept ? 1 : 0;

        data.push([x, y]);
        labels.push(label);
    }

    currentDataset = data;
    currentLabels = labels;

    return { data, labels, featureNames };
}

// Generate a dataset for a multi-class problem
function generateMultiClassDataset(numPoints = 300, numClasses = 3) {
    const data = [];
    const labels = [];

    // Create cluster centers
    const centers = [];
    for (let i = 0; i < numClasses; i++) {
        centers.push([0.2 + Math.random() * 0.6, 0.2 + Math.random() * 0.6]);
    }

    // Generate points around cluster centers
    for (let i = 0; i < numPoints; i++) {
        const classIdx = Math.floor(Math.random() * numClasses);
        const [centerX, centerY] = centers[classIdx];

        // Add random offset to center
        const x = centerX + (Math.random() - 0.5) * 0.2;
        const y = centerY + (Math.random() - 0.5) * 0.2;

        data.push([x, y]);
        labels.push(classIdx);
    }

    currentDataset = data;
    currentLabels = labels;

    return { data, labels, featureNames };
}

// Generate the Iris dataset-like structure (3 features, 3 classes)
function generateIrisLikeDataset() {
    const data = [];
    const labels = [];
    featureNames = ["sepal_length", "sepal_width", "petal_length"];

    // Create 3 clusters in 3D space
    for (let i = 0; i < 150; i++) {
        let featureVector, label;

        if (i < 50) {
            // Class 0 (like Iris-setosa)
            featureVector = [
                4.0 + Math.random() * 1.0,  // sepal length
                3.0 + Math.random() * 1.0,  // sepal width
                1.0 + Math.random() * 0.6   // petal length
            ];
            label = 0;
        } else if (i < 100) {
            // Class 1 (like Iris-versicolor)
            featureVector = [
                5.0 + Math.random() * 1.0,
                2.0 + Math.random() * 1.0,
                3.0 + Math.random() * 1.0
            ];
            label = 1;
        } else {
            // Class 2 (like Iris-virginica)
            featureVector = [
                6.0 + Math.random() * 1.0,
                2.5 + Math.random() * 1.0,
                4.5 + Math.random() * 1.0
            ];
            label = 2;
        }

        data.push(featureVector);
        labels.push(label);
    }

    currentDataset = data;
    currentLabels = labels;

    return { data, labels, featureNames };
}

// Get the current dataset
function getCurrentDataset() {
    return {
        data: currentDataset,
        labels: currentLabels,
        featureNames: featureNames
    };
}

// Split data into training and testing sets
function splitData(data, labels, testRatio = 0.2) {
    const trainData = [];
    const trainLabels = [];
    const testData = [];
    const testLabels = [];

    // Create indices array and shuffle it
    const indices = Array.from(Array(data.length).keys());
    for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    // Split data
    const testSize = Math.floor(data.length * testRatio);
    for (let i = 0; i < indices.length; i++) {
        const idx = indices[i];
        if (i < testSize) {
            testData.push(data[idx]);
            testLabels.push(labels[idx]);
        } else {
            trainData.push(data[idx]);
            trainLabels.push(labels[idx]);
        }
    }

    return {
        trainData,
        trainLabels,
        testData,
        testLabels
    };
}
