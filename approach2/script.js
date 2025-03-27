document
    .getElementById("analyze-button")
    .addEventListener("click", function () {
        const text = document.getElementById("input-text").value;
        processText(text);
    });

// Simulate NER processing with predefined entities for demonstration
function processText(text) {
    // In a real application, this would call an API or model
    // This is a simulated response for educational purposes

    // Define some common entity patterns to look for
    const entityPatterns = [
        { regex: /Neil Armstrong/g, type: "person" },
        { regex: /Buzz Aldrin/g, type: "person" },
        { regex: /NASA/g, type: "organization" },
        { regex: /Apollo 11/g, type: "organization" },
        { regex: /Mission Control/g, type: "organization" },
        { regex: /Moon/g, type: "location" },
        { regex: /Houston/g, type: "location" },
        { regex: /Texas/g, type: "location" },
        { regex: /United States/g, type: "location" },
        { regex: /July 20, 1969/g, type: "date" },
        { regex: /\$25\.4 billion/g, type: "money" },
        { regex: /Microsoft/g, type: "organization" },
        { regex: /Google/g, type: "organization" },
        { regex: /Apple/g, type: "organization" },
        { regex: /San Francisco/g, type: "location" },
        { regex: /New York/g, type: "location" },
        { regex: /Seattle/g, type: "location" },
        { regex: /January \d+, \d{4}/g, type: "date" },
        { regex: /February \d+, \d{4}/g, type: "date" },
        { regex: /March \d+, \d{4}/g, type: "date" },
        { regex: /\$\d+(\.\d+)? (million|billion)/g, type: "money" },
    ];

    // Create a copy of the original text for highlighting
    let highlightedText = text;

    // Store found entities for display in the list
    const foundEntities = {
        person: [],
        organization: [],
        location: [],
        date: [],
        money: [],
    };

    // Process each entity pattern
    entityPatterns.forEach((pattern) => {
        let match;
        while ((match = pattern.regex.exec(text)) !== null) {
            const entity = match[0];

            // Add to entity list if not already there
            if (!foundEntities[pattern.type].includes(entity)) {
                foundEntities[pattern.type].push(entity);
            }

            // Replace in the highlighted text with span-wrapped version
            highlightedText = highlightedText.replace(
                new RegExp(escapeRegExp(entity), "g"),
                `<span class="entity ${pattern.type}">${entity}</span>`
            );
        }
    });

    // Display the highlighted text
    document.getElementById("output").innerHTML = highlightedText;

    // Display the entity list
    displayEntityList(foundEntities);
}

function displayEntityList(entities) {
    const entityListElement = document.getElementById("entity-list");
    let html = "<h3>Detected Entities:</h3>";

    // Create sections for each entity type
    const types = {
        person: "Person",
        organization: "Organization",
        location: "Location",
        date: "Date",
        money: "Money",
    };

    let hasEntities = false;

    // Generate list items for each entity type
    for (const [type, label] of Object.entries(types)) {
        if (entities[type].length > 0) {
            hasEntities = true;
            html += `<h4>${label}s:</h4><ul>`;
            entities[type].forEach((entity) => {
                html += `<li class="entity-item">
                <span class="entity ${type}">${entity}</span>
             </li>`;
            });
            html += "</ul>";
        }
    }

    if (!hasEntities) {
        html += "<p>No entities detected in the provided text.</p>";
    }

    entityListElement.innerHTML = html;
}

// Helper function to escape special characters in regex
function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// Auto-run on page load with example text
window.onload = function () {
    const exampleText = document.getElementById("input-text").value;
    processText(exampleText);
};
