// Visualization module for decision tree demo

// Canvas references
let datasetCanvas, datasetCtx;
let decisionBoundaryCanvas, decisionBoundaryCtx;
let forestBoundaryCanvas, forestBoundaryCtx;

// Tree visualization state
let selectedNode = null;

// Initialize visualization components
document.addEventListener('DOMContentLoaded', function () {
    // Get canvas references
    datasetCanvas = document.getElementById('dataset-canvas');
    datasetCtx = datasetCanvas.getContext('2d');

    decisionBoundaryCanvas = document.getElementById('decision-boundary-canvas');
    decisionBoundaryCtx = decisionBoundaryCanvas.getContext('2d');

    forestBoundaryCanvas = document.getElementById('forest-boundary-canvas');
    forestBoundaryCtx = forestBoundaryCanvas.getContext('2d');

    // Add event listeners for dataset buttons
    document.getElementById('dataset-circles').addEventListener('click', function () {
        const { data, labels } = generateCircleDataset();
        visualizeDataset(data, labels);
    });

    document.getElementById('dataset-xor').addEventListener('click', function () {
        const { data, labels } = generateXORDataset();
        visualizeDataset(data, labels);
    });

    document.getElementById('dataset-spiral').addEventListener('click', function () {
        const { data, labels } = generateSpiralDataset();
        visualizeDataset(data, labels);
    });

    document.getElementById('dataset-random').addEventListener('click', function () {
        const { data, labels } = generateRandomDataset();
        visualizeDataset(data, labels);
    });

    // Draw initial simple tree and empty canvases
    drawSimpleTree();
    drawEmptyDecisionBoundary();
    drawEmptyForestBoundary();
    drawImpurityVisualization();

    // Generate initial dataset
    generateCircleDataset();
    visualizeDataset(currentDataset, currentLabels);
});

// Visualize dataset on canvas
function visualizeDataset(data, labels) {
    if (!datasetCtx) return;

    const width = datasetCanvas.width;
    const height = datasetCanvas.height;

    // Clear canvas
    datasetCtx.clearRect(0, 0, width, height);

    // Set background
    datasetCtx.fillStyle = '#f8f9fa';
    datasetCtx.fillRect(0, 0, width, height);

    // Draw grid lines
    datasetCtx.strokeStyle = '#e0e0e0';
    datasetCtx.lineWidth = 1;

    // Draw vertical grid lines
    for (let x = 0; x <= width; x += width / 10) {
        datasetCtx.beginPath();
        datasetCtx.moveTo(x, 0);
        datasetCtx.lineTo(x, height);
        datasetCtx.stroke();
    }

    // Draw horizontal grid lines
    for (let y = 0; y <= height; y += height / 10) {
        datasetCtx.beginPath();
        datasetCtx.moveTo(0, y);
        datasetCtx.lineTo(width, y);
        datasetCtx.stroke();
    }

    // Draw data points
    for (let i = 0; i < data.length; i++) {
        const [x, y] = data[i];
        const label = labels[i];

        // Map data from [0,1] range to canvas size
        const canvasX = x * width;
        const canvasY = (1 - y) * height; // Flip y-axis

        datasetCtx.beginPath();
        datasetCtx.arc(canvasX, canvasY, 4, 0, Math.PI * 2);

        // Different colors for different classes
        if (label === 0) {
            datasetCtx.fillStyle = 'rgba(52, 152, 219, 0.7)'; // Blue for class 0
        } else if (label === 1) {
            datasetCtx.fillStyle = 'rgba(231, 76, 60, 0.7)';  // Red for class 1
        } else {
            datasetCtx.fillStyle = 'rgba(46, 204, 113, 0.7)'; // Green for class 2+
        }

        datasetCtx.fill();
        datasetCtx.strokeStyle = '#333';
        datasetCtx.lineWidth = 1;
        datasetCtx.stroke();
    }
}

// Draw empty decision boundary canvas
function drawEmptyDecisionBoundary() {
    if (!decisionBoundaryCtx) return;

    const width = decisionBoundaryCanvas.width;
    const height = decisionBoundaryCanvas.height;

    decisionBoundaryCtx.clearRect(0, 0, width, height);
    decisionBoundaryCtx.fillStyle = '#f8f9fa';
    decisionBoundaryCtx.fillRect(0, 0, width, height);

    // Add instructions
    decisionBoundaryCtx.fillStyle = '#777';
    decisionBoundaryCtx.font = '14px Arial';
    decisionBoundaryCtx.textAlign = 'center';
    decisionBoundaryCtx.fillText('Select a dataset and build a tree', width / 2, height / 2 - 20);
    decisionBoundaryCtx.fillText('to see the decision boundary', width / 2, height / 2 + 20);
}

// Draw empty forest boundary canvas
function drawEmptyForestBoundary() {
    if (!forestBoundaryCtx) return;

    const width = forestBoundaryCanvas.width;
    const height = forestBoundaryCanvas.height;

    forestBoundaryCtx.clearRect(0, 0, width, height);
    forestBoundaryCtx.fillStyle = '#f8f9fa';
    forestBoundaryCtx.fillRect(0, 0, width, height);

    // Add instructions
    forestBoundaryCtx.fillStyle = '#777';
    forestBoundaryCtx.font = '14px Arial';
    forestBoundaryCtx.textAlign = 'center';
    forestBoundaryCtx.fillText('Build a random forest', width / 2, height / 2 - 20);
    forestBoundaryCtx.fillText('to see the decision boundary', width / 2, height / 2 + 20);
}

// Draw decision boundary
function drawDecisionBoundary(treeModel, data, labels) {
    if (!decisionBoundaryCtx || !treeModel) return;

    const width = decisionBoundaryCanvas.width;
    const height = decisionBoundaryCanvas.height;

    // Clear canvas
    decisionBoundaryCtx.clearRect(0, 0, width, height);

    // Create a grid of points to predict
    const resolution = 80; // Number of points along each axis
    const gridSize = width / resolution;

    // Draw decision regions
    for (let y = 0; y < resolution; y++) {
        for (let x = 0; x < resolution; x++) {
            // Convert grid coordinates to feature space [0,1]
            const featureX = x / resolution;
            const featureY = y / resolution;

            // Predict class for this point
            const predictedClass = predictWithTree(treeModel, [featureX, featureY]);

            // Set the fill color based on the predicted class
            if (predictedClass === 0) {
                decisionBoundaryCtx.fillStyle = 'rgba(52, 152, 219, 0.3)'; // Light blue for class 0
            } else if (predictedClass === 1) {
                decisionBoundaryCtx.fillStyle = 'rgba(231, 76, 60, 0.3)';  // Light red for class 1
            } else {
                decisionBoundaryCtx.fillStyle = 'rgba(46, 204, 113, 0.3)'; // Light green for class 2+
            }

            // Draw grid cell
            decisionBoundaryCtx.fillRect(x * gridSize, y * gridSize, gridSize, gridSize);
        }
    }

    // Draw data points
    for (let i = 0; i < data.length; i++) {
        const [x, y] = data[i];
        const label = labels[i];

        // Map data from [0,1] range to canvas size
        const canvasX = x * width;
        const canvasY = (1 - y) * height; // Flip y-axis

        decisionBoundaryCtx.beginPath();
        decisionBoundaryCtx.arc(canvasX, canvasY, 4, 0, Math.PI * 2);

        // Different colors for different classes
        if (label === 0) {
            decisionBoundaryCtx.fillStyle = 'rgba(52, 152, 219, 0.9)'; // Blue for class 0
        } else if (label === 1) {
            decisionBoundaryCtx.fillStyle = 'rgba(231, 76, 60, 0.9)';  // Red for class 1
        } else {
            decisionBoundaryCtx.fillStyle = 'rgba(46, 204, 113, 0.9)'; // Green for class 2+
        }

        decisionBoundaryCtx.fill();
        decisionBoundaryCtx.strokeStyle = '#333';
        decisionBoundaryCtx.lineWidth = 1;
        decisionBoundaryCtx.stroke();
    }
}

// Draw forest decision boundary
function drawForestBoundary(forestModel, data, labels) {
    if (!forestBoundaryCtx || !forestModel) return;

    const width = forestBoundaryCanvas.width;
    const height = forestBoundaryCanvas.height;

    // Clear canvas
    forestBoundaryCtx.clearRect(0, 0, width, height);

    // Create a grid of points to predict
    const resolution = 80; // Number of points along each axis
    const gridSize = width / resolution;

    // Draw decision regions
    for (let y = 0; y < resolution; y++) {
        for (let x = 0; x < resolution; x++) {
            // Convert grid coordinates to feature space [0,1]
            const featureX = x / resolution;
            const featureY = y / resolution;

            // Predict class for this point
            const predictedClass = predictWithForest(forestModel, [featureX, featureY]);

            // Set the fill color based on the predicted class
            if (predictedClass === 0) {
                forestBoundaryCtx.fillStyle = 'rgba(52, 152, 219, 0.3)'; // Light blue for class 0
            } else if (predictedClass === 1) {
                forestBoundaryCtx.fillStyle = 'rgba(231, 76, 60, 0.3)';  // Light red for class 1
            } else {
                forestBoundaryCtx.fillStyle = 'rgba(46, 204, 113, 0.3)'; // Light green for class 2+
            }

            // Draw grid cell
            forestBoundaryCtx.fillRect(x * gridSize, y * gridSize, gridSize, gridSize);
        }
    }

    // Draw data points
    for (let i = 0; i < data.length; i++) {
        const [x, y] = data[i];
        const label = labels[i];

        // Map data from [0,1] range to canvas size
        const canvasX = x * width;
        const canvasY = (1 - y) * height; // Flip y-axis

        forestBoundaryCtx.beginPath();
        forestBoundaryCtx.arc(canvasX, canvasY, 4, 0, Math.PI * 2);

        // Different colors for different classes
        if (label === 0) {
            forestBoundaryCtx.fillStyle = 'rgba(52, 152, 219, 0.9)'; // Blue for class 0
        } else if (label === 1) {
            forestBoundaryCtx.fillStyle = 'rgba(231, 76, 60, 0.9)';  // Red for class 1
        } else {
            forestBoundaryCtx.fillStyle = 'rgba(46, 204, 113, 0.9)'; // Green for class 2+
        }

        forestBoundaryCtx.fill();
        forestBoundaryCtx.strokeStyle = '#333';
        forestBoundaryCtx.lineWidth = 1;
        forestBoundaryCtx.stroke();
    }
}

// Draw a simple tree diagram for the intro section
function drawSimpleTree() {
    // Example simple tree data
    const simpleTreeData = {
        name: "Outlook",
        children: [
            {
                name: "Sunny",
                children: [
                    {
                        name: "Humidity",
                        children: [
                            { name: "High", value: "Don't Play", isLeaf: true },
                            { name: "Normal", value: "Play", isLeaf: true }
                        ]
                    }
                ]
            },
            {
                name: "Overcast",
                value: "Play",
                isLeaf: true
            },
            {
                name: "Rain",
                children: [
                    {
                        name: "Wind",
                        children: [
                            { name: "Strong", value: "Don't Play", isLeaf: true },
                            { name: "Weak", value: "Play", isLeaf: true }
                        ]
                    }
                ]
            }
        ]
    };

    // Set up the D3 tree layout
    const width = 500;
    const height = 220;
    const margin = { top: 20, right: 90, bottom: 30, left: 90 };

    // Append the SVG object to the tree diagram container
    const svg = d3.select("#tree-diagram-simple")
        .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    // Declare a tree layout
    const treeLayout = d3.tree().size([height, width - 160]);

    // Assigns parent, children, height, depth
    const root = d3.hierarchy(simpleTreeData);
    treeLayout(root);

    // Add links between nodes
    svg.selectAll(".link")
        .data(root.links())
        .enter()
        .append("path")
        .attr("class", "link")
        .attr("d", d3.linkHorizontal()
            .x(d => d.y)
            .y(d => d.x))
        .style("fill", "none")
        .style("stroke", "#ccc")
        .style("stroke-width", 2);

    // Add nodes
    const nodes = svg.selectAll(".node")
        .data(root.descendants())
        .enter()
        .append("g")
        .attr("class", d => "node" + (d.children ? " node--internal" : " node--leaf"))
        .attr("transform", d => `translate(${d.y},${d.x})`);

    // Add node circles
    nodes.append("circle")
        .attr("r", 10)
        .style("fill", d => d.data.isLeaf ? "#e74c3c" : "#3498db")
        .style("stroke", "#2c3e50")
        .style("stroke-width", 1);

    // Add node labels
    nodes.append("text")
        .attr("dy", ".35em")
        .attr("x", d => d.children ? -15 : 15)
        .style("text-anchor", d => d.children ? "end" : "start")
        .style("font-size", "12px")
        .style("font-family", "Arial")
        .style("fill", "#333")
        .text(d => d.data.isLeaf ? d.data.value : d.data.name);

    // Add the edge labels (conditions)
    nodes.filter(d => d.depth > 0)
        .append("text")
        .attr("dy", -10)
        .attr("x", -15)
        .style("text-anchor", "end")
        .style("font-size", "10px")
        .style("font-family", "Arial")
        .style("fill", "#777")
        .text(d => d.data.name);
}

// Visualize tree structure
function visualizeTreeStructure(treeData) {
    // Set dimensions and margins for the diagram
    const width = 600;
    const height = 400;
    const margin = { top: 20, right: 90, bottom: 30, left: 90 };

    // Clear previous tree
    d3.select("#tree-structure").html("");

    // Append the SVG object
    const svg = d3.select("#tree-structure")
        .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    // Create hierarchy and tree layout
    const root = d3.hierarchy(treeData);
    const treeLayout = d3.tree().size([height, width - 160]);
    treeLayout(root);

    // Add links
    svg.selectAll(".link")
        .data(root.links())
        .enter()
        .append("path")
        .attr("class", "link")
        .attr("d", d3.linkHorizontal()
            .x(d => d.y)
            .y(d => d.x))
        .style("fill", "none")
        .style("stroke", "#ccc")
        .style("stroke-width", 2);

    // Add nodes
    const nodes = svg.selectAll(".node")
        .data(root.descendants())
        .enter()
        .append("g")
        .attr("class", d => "node" + (d.children ? " node--internal" : " node--leaf"))
        .attr("transform", d => `translate(${d.y},${d.x})`)
        .on("click", function (event, d) {
            handleNodeClick(event, d);
        });

    // Add node circles
    nodes.append("circle")
        .attr("r", d => d.data.isLeaf ? 12 : 10)
        .style("fill", d => d.data.isLeaf ? "#e74c3c" : "#3498db")
        .style("stroke", "#2c3e50")
        .style("stroke-width", 1);

    // Add node labels
    nodes.append("text")
        .attr("dy", ".35em")
        .attr("x", d => d.children ? -15 : 15)
        .style("text-anchor", d => d.children ? "end" : "start")
        .style("font-size", "12px")
        .style("font-family", "Arial")
        .style("fill", "#333")
        .text(d => {
            if (d.data.isLeaf) {
                return `Class: ${d.data.prediction}`;
            } else {
                return `${d.data.feature}: ${d.data.threshold.toFixed(2)}`;
            }
        });

    // Add sample counts
    nodes.append("text")
        .attr("dy", "1.75em")
        .attr("x", d => d.children ? -15 : 15)
        .style("text-anchor", d => d.children ? "end" : "start")
        .style("font-size", "10px")
        .style("font-family", "Arial")
        .style("fill", "#777")
        .text(d => `n=${d.data.samples}`);
}

// Handle node click
function handleNodeClick(event, d) {
    selectedNode = d;

    // Update node details display
    updateNodeDetails(d);

    // Highlight the selected node
    d3.selectAll("circle")
        .style("stroke", "#2c3e50")
        .style("stroke-width", 1);

    d3.select(event.currentTarget).select("circle")
        .style("stroke", "#f39c12")
        .style("stroke-width", 3);
}

// Update node details display
function updateNodeDetails(node) {
    const detailsDiv = document.getElementById("node-details");

    let detailsHTML = `<div class="node-details-card">`;

    if (node.data.isLeaf) {
        detailsHTML += `
            <h4>Leaf Node</h4>
            <p><strong>Prediction:</strong> Class ${node.data.prediction}</p>
            <p><strong>Samples:</strong> ${node.data.samples}</p>
            <p><strong>Class Distribution:</strong></p>
            <div class="distribution-bar">
        `;

        // Calculate distribution percentages
        const total = node.data.samples;
        const classDistribution = node.data.classDistribution || {};

        for (const classLabel in classDistribution) {
            const count = classDistribution[classLabel];
            const percentage = (count / total * 100).toFixed(1);

            let color;
            if (parseInt(classLabel) === 0) color = "#3498db";
            else if (parseInt(classLabel) === 1) color = "#e74c3c";
            else color = "#2ecc71";

            detailsHTML += `
                <div class="dist-segment" style="width: ${percentage}%; background-color: ${color};" 
                     title="Class ${classLabel}: ${count} samples (${percentage}%)"></div>
            `;
        }

        detailsHTML += `</div>`;

        // Add class distribution as text
        detailsHTML += `<div class="dist-labels">`;
        for (const classLabel in classDistribution) {
            const count = classDistribution[classLabel];
            const percentage = (count / total * 100).toFixed(1);

            detailsHTML += `<div>Class ${classLabel}: ${count} (${percentage}%)</div>`;
        }
        detailsHTML += `</div>`;

    } else {
        detailsHTML += `
            <h4>Decision Node</h4>
            <p><strong>Split Feature:</strong> ${node.data.feature}</p>
            <p><strong>Threshold:</strong> ${node.data.threshold.toFixed(4)}</p>
            <p><strong>Samples:</strong> ${node.data.samples}</p>
            <p><strong>Impurity:</strong> ${node.data.impurity.toFixed(4)}</p>
            <p><strong>Split Criterion:</strong> ${node.data.criterion || "gini"}</p>
            
            <div class="split-info">
                <div class="split-direction">
                    <div class="arrow left">←</div>
                    <div>Left Branch (≤ threshold)</div>
                    <div>Samples: ${node.data.leftSamples || '?'}</div>
                </div>
                <div class="split-direction">
                    <div class="arrow right">→</div>
                    <div>Right Branch (> threshold)</div>
                    <div>Samples: ${node.data.rightSamples || '?'}</div>
                </div>
            </div>
        `;

        // Class distribution
        detailsHTML += `<p><strong>Class Distribution:</strong></p>
            <div class="distribution-bar">`;

        // Calculate distribution percentages
        const total = node.data.samples;
        const classDistribution = node.data.classDistribution || {};

        for (const classLabel in classDistribution) {
            const count = classDistribution[classLabel];
            const percentage = (count / total * 100).toFixed(1);

            let color;
            if (parseInt(classLabel) === 0) color = "#3498db";
            else if (parseInt(classLabel) === 1) color = "#e74c3c";
            else color = "#2ecc71";

            detailsHTML += `
                <div class="dist-segment" style="width: ${percentage}%; background-color: ${color};" 
                     title="Class ${classLabel}: ${count} samples (${percentage}%)"></div>
            `;
        }

        detailsHTML += `</div>`;

        // Add class distribution as text
        detailsHTML += `<div class="dist-labels">`;
        for (const classLabel in classDistribution) {
            const count = classDistribution[classLabel];
            const percentage = (count / total * 100).toFixed(1);

            detailsHTML += `<div>Class ${classLabel}: ${count} (${percentage}%)</div>`;
        }
        detailsHTML += `</div>`;
    }

    detailsHTML += `</div>`;
    detailsDiv.innerHTML = detailsHTML;
}

// Visualize a sample of trees from the forest
function visualizeForestSample(forestData, sampleSize = 3) {
    const container = document.getElementById('forest-sample-trees');
    container.innerHTML = '';

    // Choose a random subset of trees to display
    const treesToShow = Math.min(sampleSize, forestData.trees.length);

    // Create a container for each tree
    for (let i = 0; i < treesToShow; i++) {
        const treeContainer = document.createElement('div');
        treeContainer.className = 'forest-tree-container';
        treeContainer.innerHTML = `<h4>Tree ${i + 1}</h4>`;

        // Create SVG container for this tree
        const treeDiv = document.createElement('div');
        treeDiv.id = `forest-tree-${i}`;
        treeDiv.className = 'forest-tree';
        treeContainer.appendChild(treeDiv);

        container.appendChild(treeContainer);

        // Now visualize this tree
        visualizeForestTree(forestData.trees[i], `forest-tree-${i}`);
    }
}

// Visualize a single tree from the forest
function visualizeForestTree(treeData, containerId) {
    // Set dimensions and margins for the diagram - smaller than main tree
    const width = 400;
    const height = 300;
    const margin = { top: 20, right: 90, bottom: 30, left: 90 };

    // Append the SVG object
    const svg = d3.select(`#${containerId}`)
        .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    // Create hierarchy and tree layout
    const root = d3.hierarchy(treeData);
    const treeLayout = d3.tree().size([height, width - 160]);
    treeLayout(root);

    // Add links
    svg.selectAll(".link")
        .data(root.links())
        .enter()
        .append("path")
        .attr("class", "link")
        .attr("d", d3.linkHorizontal()
            .x(d => d.y)
            .y(d => d.x))
        .style("fill", "none")
        .style("stroke", "#ccc")
        .style("stroke-width", 1.5);

    // Add nodes
    const nodes = svg.selectAll(".node")
        .data(root.descendants())
        .enter()
        .append("g")
        .attr("class", d => "node" + (d.children ? " node--internal" : " node--leaf"))
        .attr("transform", d => `translate(${d.y},${d.x})`);

    // Add node circles
    nodes.append("circle")
        .attr("r", d => d.data.isLeaf ? 8 : 6)
        .style("fill", d => d.data.isLeaf ? "#e74c3c" : "#3498db")
        .style("stroke", "#2c3e50")
        .style("stroke-width", 1);

    // Add node labels - simplified for forest trees (smaller)
    nodes.append("text")
        .attr("dy", ".35em")
        .attr("x", d => d.children ? -10 : 10)
        .style("text-anchor", d => d.children ? "end" : "start")
        .style("font-size", "10px")
        .style("font-family", "Arial")
        .style("fill", "#333")
        .text(d => {
            if (d.data.isLeaf) {
                return `${d.data.prediction}`;
            } else {
                return `${d.data.feature.substr(0, 1)}:${d.data.threshold.toFixed(1)}`;
            }
        });
}

// Setup feature inputs for prediction
function setupFeatureInputs(featureNames) {
    const container = document.getElementById('feature-inputs');
    container.innerHTML = '';

    for (let i = 0; i < featureNames.length; i++) {
        const featureName = featureNames[i];

        const inputDiv = document.createElement('div');
        inputDiv.className = 'feature-input';

        const label = document.createElement('label');
        label.textContent = `${featureName}:`;
        label.htmlFor = `feature-${i}`;

        const input = document.createElement('input');
        input.type = 'number';
        input.id = `feature-${i}`;
        input.min = '0';
        input.max = '1';
        input.step = '0.01';
        input.value = '0.5';

        inputDiv.appendChild(label);
        inputDiv.appendChild(input);
        container.appendChild(inputDiv);
    }
}

// Visualize prediction path
function visualizePredictionPath(tree, sampleFeatures, path) {
    const pathDiv = document.getElementById('prediction-path');

    let pathHTML = '<div class="decision-path">';

    for (let i = 0; i < path.length; i++) {
        const node = path[i];

        if (i === path.length - 1) {
            // Leaf node (final prediction)
            pathHTML += `
                <div class="path-node leaf">
                    <div class="node-icon">🏁</div>
                    <div class="node-content">
                        <div class="node-title">Final Prediction: Class ${node.prediction}</div>
                        <div class="node-details">This is a leaf node with ${node.samples} samples</div>
                    </div>
                </div>
            `;
        } else {
            // Decision node
            const featureValue = sampleFeatures[node.featureIndex];
            const direction = featureValue <= node.threshold ? "left" : "right";
            const nextNode = path[i + 1];

            pathHTML += `
                <div class="path-node">
                    <div class="node-icon">❓</div>
                    <div class="node-content">
                        <div class="node-title">Check ${node.feature} (${featureValue.toFixed(2)})</div>
                        <div class="node-details">
                            Is ${node.feature} ≤ ${node.threshold.toFixed(2)}? 
                            <strong>${featureValue <= node.threshold ? "Yes" : "No"}</strong>
                        </div>
                        <div class="node-action">Go ${direction} to ${nextNode.isLeaf ? 'leaf node' : 'next decision'}</div>
                    </div>
                </div>
                <div class="path-arrow">⬇️</div>
            `;
        }
    }

    pathHTML += '</div>';
    pathDiv.innerHTML = pathHTML;
}

// Draw impurity visualization
function drawImpurityVisualization() {
    const container = document.getElementById('impurity-visualization');
    container.innerHTML = `
        <h4>Impurity Measures Comparison</h4>
        <canvas id="impurity-canvas" width="600" height="300"></canvas>
        <div class="impurity-legend">
            <div class="legend-item">
                <div class="legend-color" style="background-color: #3498db;"></div>
                <div>Gini Impurity</div>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #e74c3c;"></div>
                <div>Entropy</div>
            </div>
        </div>
    `;

    // Get canvas context
    const canvas = document.getElementById('impurity-canvas');
    const ctx = canvas.getContext('2d');

    // Draw grid and axes
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(0, 0, width, height);

    // Draw grid lines
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;

    // Vertical grid lines
    for (let x = 0; x <= width; x += width / 10) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height - 40);
        ctx.stroke();
    }

    // Horizontal grid lines
    for (let y = 0; y <= height - 40; y += (height - 40) / 10) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
    }

    // Draw x and y axes
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;

    // x-axis
    ctx.beginPath();
    ctx.moveTo(0, height - 40);
    ctx.lineTo(width, height - 40);
    ctx.stroke();

    // y-axis
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(0, height - 40);
    ctx.stroke();

    // Label axes
    ctx.fillStyle = '#333';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';

    // x-axis labels
    for (let i = 0; i <= 10; i++) {
        const x = i * width / 10;
        const pValue = i / 10;
        ctx.fillText(pValue.toFixed(1), x, height - 20);
    }

    // Add axis titles
    ctx.font = '14px Arial';
    ctx.fillText('Probability of Class 1 (p)', width / 2, height - 5);

    ctx.save();
    ctx.translate(15, height / 2 - 20);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('Impurity Value', 0, 0);
    ctx.restore();

    // Draw Gini impurity curve
    ctx.beginPath();
    ctx.strokeStyle = '#3498db';
    ctx.lineWidth = 3;

    for (let x = 0; x <= width; x++) {
        const p = x / width;
        const gini = 2 * p * (1 - p); // Gini = 1 - (p^2 + (1-p)^2) = 2p(1-p)
        const y = (height - 40) * (1 - gini); // Scale and invert

        if (x === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    ctx.stroke();

    // Draw Entropy curve
    ctx.beginPath();
    ctx.strokeStyle = '#e74c3c';
    ctx.lineWidth = 3;

    for (let x = 0; x <= width; x++) {
        const p = x / width;
        let entropy = 0;

        if (p > 0 && p < 1) {
            // Avoid log(0) which is undefined
            entropy = -p * Math.log2(p) - (1 - p) * Math.log2(1 - p);
        }

        // Normalize entropy to [0,1] range (max entropy is 1 at p=0.5)
        const normalizedEntropy = entropy;
        const y = (height - 40) * (1 - normalizedEntropy);

        if (x === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    ctx.stroke();
}

// Draw feature importance chart
function drawFeatureImportanceChart(featureNames, importances) {
    const container = document.getElementById('feature-importance-chart');
    container.innerHTML = '';

    // Create a bar chart
    const chart = document.createElement('div');
    chart.className = 'importance-chart';

    // Sort features by importance
    const featureImportances = featureNames.map((name, index) => ({
        name,
        importance: importances[index]
    }));

    featureImportances.sort((a, b) => b.importance - a.importance);

    // Create bars for each feature
    for (const feature of featureImportances) {
        const barContainer = document.createElement('div');
        barContainer.className = 'importance-bar-container';

        const label = document.createElement('div');
        label.className = 'importance-label';
        label.textContent = feature.name;

        const barWrapper = document.createElement('div');
        barWrapper.className = 'importance-bar-wrapper';

        const bar = document.createElement('div');
        bar.className = 'importance-bar';
        bar.style.width = `${feature.importance * 100}%`;

        const value = document.createElement('div');
        value.className = 'importance-value';
        value.textContent = feature.importance.toFixed(3);

        barWrapper.appendChild(bar);
        barContainer.appendChild(label);
        barContainer.appendChild(barWrapper);
        barContainer.appendChild(value);
        chart.appendChild(barContainer);
    }

    container.appendChild(chart);
}

// Draw performance comparison chart
function drawPerformanceComparison(treeAccuracy, forestAccuracy) {
    const container = document.getElementById('performance-comparison');
    container.innerHTML = `
        <h4>Accuracy Comparison</h4>
        <div class="performance-chart">
            <div class="perf-model">
                <div class="perf-label">Decision Tree</div>
                <div class="perf-bar-wrapper">
                    <div class="perf-bar" style="width: ${treeAccuracy * 100}%"></div>
                </div>
                <div class="perf-value">${(treeAccuracy * 100).toFixed(1)}%</div>
            </div>
            <div class="perf-model">
                <div class="perf-label">Random Forest</div>
                <div class="perf-bar-wrapper">
                    <div class="perf-bar" style="width: ${forestAccuracy * 100}%"></div>
                </div>
                <div class="perf-value">${(forestAccuracy * 100).toFixed(1)}%</div>
            </div>
        </div>
    `;
}
